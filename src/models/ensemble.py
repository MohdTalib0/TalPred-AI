"""Ensemble model trainer: XGBoost + LightGBM averaged predictions.

Reduces model-selection risk by combining two gradient boosting frameworks
with slightly different inductive biases (split strategy, regularization,
histogram construction). Expected IC improvement: 2-5% over single model.

Usage:
    from src.models.ensemble import train_ensemble

    result = train_ensemble(df, feature_profile="cross_sectional_alpha")
    model = result["model"]  # EnsembleModel with .predict_proba()
"""

import logging
from datetime import date
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from src.ml.tracking import configure_mlflow
from src.models.trainer import (
    DEFAULT_PARAMS,
    FEATURE_PROFILES,
    prepare_features,
)

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed — ensemble will use XGBoost only")


LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}


class EnsembleModel:
    """Wraps multiple models with averaged predict_proba.

    Compatible with the XGBClassifier interface used by the rest of
    the pipeline (predict, predict_proba, feature_importances_).
    """

    def __init__(
        self,
        models: list,
        weights: list[float] | None = None,
        feature_names: list[str] | None = None,
    ):
        if not models:
            raise ValueError(
                "EnsembleModel requires at least one model. "
                "Check that XGBoost/LightGBM training succeeded."
            )
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self._feature_names: list[str] | None = feature_names

    @staticmethod
    def _ensure_2d(proba: np.ndarray, n_samples: int) -> np.ndarray:
        """Normalize predict_proba output to shape (n, 2).

        LightGBM can return (n,) in edge cases; XGBoost always returns
        (n, 2). This guard makes aggregation safe regardless of backend.
        """
        if proba.ndim == 1:
            return np.column_stack([1.0 - proba, proba])
        if proba.ndim == 2 and proba.shape[1] == 1:
            return np.column_stack([1.0 - proba[:, 0], proba[:, 0]])
        return proba

    def predict_proba(self, X) -> np.ndarray:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        avg = np.zeros((n_samples, 2), dtype=np.float64)
        for model, w in zip(self.models, self.weights):
            p = self._ensure_2d(model.predict_proba(X), n_samples)
            avg += p * w
        return avg

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        parts = []
        for model, w in zip(self.models, self.weights):
            imp = model.feature_importances_.copy()
            total = imp.sum()
            if total > 0:
                imp = imp / total
            parts.append(imp * w)
        return np.array(parts).sum(axis=0)

    def get_booster(self):
        """Return a booster-like object exposing ``feature_names``.

        Works for pure-XGBoost ensembles (delegates to models[0]) and
        for mixed XGBoost/LightGBM ensembles (returns a lightweight
        namespace backed by ``self._feature_names``).
        """
        if self._feature_names is not None:
            return SimpleNamespace(feature_names=self._feature_names)
        try:
            return self.models[0].get_booster()
        except AttributeError:
            return SimpleNamespace(feature_names=None)

    @property
    def n_models(self) -> int:
        return len(self.models)


def train_ensemble(
    df: pd.DataFrame,
    xgb_params: dict | None = None,
    lgb_params: dict | None = None,
    feature_profile: str = "all_features",
    xgb_seeds: list[int] | None = None,
    lgb_seeds: list[int] | None = None,
) -> dict:
    """Train XGBoost + LightGBM ensemble with multi-seed averaging.

    Default: 3 XGBoost models (seeds 42,43,44) + 2 LightGBM models
    (seeds 42,43), averaged with equal weight.

    Returns dict matching train_baseline interface.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

    xgb_params = xgb_params or DEFAULT_PARAMS.copy()
    lgb_params = lgb_params or LIGHTGBM_PARAMS.copy()
    xgb_seeds = xgb_seeds or [42, 43, 44]
    lgb_seeds = lgb_seeds or [42, 43]

    split_idx = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train, train_medians = prepare_features(
        df_train, feature_profile=feature_profile,
    )
    X_val, y_val, _ = prepare_features(
        df_val, fill_medians=train_medians, feature_profile=feature_profile,
    )

    common_cols = sorted([c for c in X_train.columns if c in X_val.columns])
    X_train, X_val = X_train[common_cols], X_val[common_cols]

    logger.info(
        f"Ensemble training: {len(X_train)} train, {len(X_val)} val, "
        f"{len(common_cols)} features (sorted for cross-model consistency)"
    )

    models = []
    model_names = []

    # XGBoost models
    for seed in xgb_seeds:
        p = xgb_params.copy()
        p["random_state"] = seed
        m = xgb.XGBClassifier(**p)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models.append(m)
        model_names.append(f"xgb_seed{seed}")
        prob = m.predict_proba(X_val)[:, 1]
        ic = float(pd.Series(prob).corr(
            pd.Series(y_val.values).astype(float), method="spearman",
        ))
        acc = float(accuracy_score(y_val, m.predict(X_val)))
        logger.info(f"  {model_names[-1]}: acc={acc:.4f}, IC={ic:.4f}")

    # LightGBM models
    if HAS_LIGHTGBM:
        for seed in lgb_seeds:
            p = lgb_params.copy()
            p["random_state"] = seed
            m = lgb.LGBMClassifier(**p)
            m.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(period=0)],
            )
            models.append(m)
            model_names.append(f"lgb_seed{seed}")
            prob = m.predict_proba(X_val)[:, 1]
            ic = float(pd.Series(prob).corr(
                pd.Series(y_val.values).astype(float), method="spearman",
            ))
            acc = float(accuracy_score(y_val, m.predict(X_val)))
            logger.info(f"  {model_names[-1]}: acc={acc:.4f}, IC={ic:.4f}")
    else:
        logger.warning("LightGBM not available — ensemble uses XGBoost only")

    if not models:
        raise RuntimeError(
            "No models were trained successfully. Cannot build ensemble. "
            "Check training data and parameters."
        )

    ensemble = EnsembleModel(models, feature_names=list(common_cols))

    y_prob = ensemble.predict_proba(X_val)[:, 1]
    y_pred = ensemble.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
        "auc_roc": float(roc_auc_score(y_val, y_prob)),
        "log_loss": float(log_loss(y_val, y_prob)),
        "val_up_pct": float(y_val.mean()),
        "n_models": ensemble.n_models,
        "model_names": model_names,
    }

    val_ic = float(pd.Series(y_prob).corr(
        pd.Series(y_val.values).astype(float), method="spearman",
    ))
    metrics["val_ic"] = val_ic

    importance = dict(zip(common_cols, ensemble.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    logger.info(f"\nEnsemble results ({ensemble.n_models} models):")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    logger.info(f"  IC:       {val_ic:.4f}")
    logger.info(f"  Models:   {model_names}")

    train_start = str(df["target_session_date"].min()) if "target_session_date" in df.columns else None
    train_end = str(df["target_session_date"].max()) if "target_session_date" in df.columns else None

    # Log to MLflow so ensemble models get a valid run_id for promotion
    mlflow_run_id = None
    try:
        configure_mlflow()
        with mlflow.start_run(run_name=f"ensemble_{feature_profile}") as run:
            mlflow_run_id = run.info.run_id
            mlflow.log_params({
                "model_mode": "ensemble",
                "feature_profile": feature_profile,
                "n_models": ensemble.n_models,
                "component_models": ",".join(model_names),
            })
            mlflow.log_metrics({
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            })
            mlflow.xgboost.log_model(models[0], "model")
    except Exception:
        logger.warning("MLflow logging failed for ensemble", exc_info=True)

    return {
        "model": ensemble,
        "model_mode": "ensemble",
        "metrics": metrics,
        "run_id": mlflow_run_id,
        "feature_columns": list(common_cols),
        "importance": importance,
        "train_medians": train_medians,
        "training_window": (train_start, train_end),
        "feature_profile": feature_profile,
        "component_models": model_names,
    }
