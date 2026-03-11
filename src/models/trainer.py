"""Baseline XGBoost classifier trainer (ML-205 + ML-301).

Trains a directional prediction model using leakage-safe features.
Integrates with MLflow via standardized tracking (ML-301).
"""

import logging
from datetime import date

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from src.ml.tracking import configure_mlflow, set_standard_tags

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "rsi_14", "momentum_5d", "momentum_10d",
    "rolling_return_5d", "rolling_return_20d", "rolling_volatility_20d",
    "macd", "macd_signal",
    "sector_return_1d", "sector_return_5d",
    "benchmark_relative_return_1d",
    "vix_level",
]

CATEGORICAL_FEATURES = ["regime_label"]

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}


def prepare_features(
    df: pd.DataFrame,
    fill_medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare feature matrix X and target y from training dataset.

    Args:
        df: DataFrame with features + direction column
        fill_medians: If provided, use these medians for NaN filling instead
                      of computing from df (prevents val leaking into train).

    Returns (X, y, medians) where medians can be passed to subsequent calls.
    """
    df = df.copy()

    if "regime_label" in df.columns:
        regime_dummies = pd.get_dummies(df["regime_label"], prefix="regime", dtype=float)
        df = pd.concat([df, regime_dummies], axis=1)
        extra_cols = list(regime_dummies.columns)
    else:
        extra_cols = []

    feature_cols = FEATURE_COLS + extra_cols
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df["direction"].copy()

    medians = fill_medians if fill_medians is not None else X.median()
    X = X.fillna(medians)

    return X, y, medians


def train_baseline(
    df: pd.DataFrame,
    params: dict | None = None,
    experiment_name: str = "talpred-baseline",
    run_name: str | None = None,
    dataset_version: str | None = None,
) -> dict:
    """Train baseline XGBoost and log to MLflow.

    Returns dict with model, metrics, run_id, and training metadata.
    """
    params = params or DEFAULT_PARAMS.copy()
    configure_mlflow()

    split_idx = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train, train_medians = prepare_features(df_train)
    X_val, y_val, _ = prepare_features(df_val, fill_medians=train_medians)

    common_cols = [c for c in X_train.columns if c in X_val.columns]
    X_train, X_val = X_train[common_cols], X_val[common_cols]

    train_start = str(df["target_session_date"].min()) if "target_session_date" in df.columns else None
    train_end = str(df["target_session_date"].max()) if "target_session_date" in df.columns else None

    logger.info(f"Training: {len(X_train)} rows, Validation: {len(X_val)} rows")
    logger.info(f"Features: {list(X_train.columns)}")
    logger.info(f"Target distribution - train: {y_train.mean():.3f}, val: {y_val.mean():.3f}")

    try:
        mlflow.set_experiment(experiment_name)
    except Exception:
        logger.warning("Remote MLflow setup failed, falling back to local tracking")
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        set_standard_tags(
            dataset_version=dataset_version,
            training_window_start=train_start,
            training_window_end=train_end,
        )

        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("feature_columns", str(list(X_train.columns)))
        if dataset_version:
            mlflow.log_param("dataset_version", dataset_version)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "auc_roc": roc_auc_score(y_val, y_prob),
            "log_loss": log_loss(y_val, y_prob),
            "val_up_pct": y_val.mean(),
        }

        mlflow.log_metrics(metrics)

        importance = dict(zip(X_train.columns, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, imp in top_features:
            mlflow.log_metric(f"importance_{feat}", float(imp))

        mlflow.xgboost.log_model(model, "model")

        logger.info(f"Baseline results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
        logger.info(f"  Top features: {top_features[:5]}")

        if metrics["accuracy"] > 0.65:
            logger.warning(
                f"LEAKAGE ALERT: Accuracy {metrics['accuracy']:.4f} > 0.65 threshold. "
                "Triggering audit."
            )

        return {
            "model": model,
            "metrics": metrics,
            "run_id": run.info.run_id,
            "feature_columns": list(X_train.columns),
            "importance": importance,
            "train_medians": train_medians,
            "training_window": (train_start, train_end),
        }
