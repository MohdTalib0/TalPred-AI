"""Walk-forward backtest pipeline (ML-206).

Implements expanding-window walk-forward validation to evaluate model
performance without look-ahead bias.
"""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.trainer import DEFAULT_PARAMS, prepare_features

logger = logging.getLogger(__name__)


def walk_forward_backtest(
    df: pd.DataFrame,
    min_train_days: int = 252,
    step_days: int = 21,
    params: dict | None = None,
) -> dict:
    """Run walk-forward expanding-window backtest.

    Args:
        df: Full dataset with features + direction + target_session_date
        min_train_days: Minimum trading days before first prediction
        step_days: Number of days per evaluation step
        params: XGBoost parameters

    Returns dict with per-step metrics and aggregate results.
    """
    params = params or DEFAULT_PARAMS.copy()
    df = df.sort_values("target_session_date").reset_index(drop=True)

    dates = sorted(df["target_session_date"].unique())
    if len(dates) < min_train_days + step_days:
        logger.warning(f"Not enough data for backtest: {len(dates)} dates, need {min_train_days + step_days}")
        return {"error": "insufficient data"}

    step_results = []
    all_predictions = []

    start_idx = min_train_days
    while start_idx < len(dates):
        end_idx = min(start_idx + step_days, len(dates))

        train_end = dates[start_idx - 1]
        test_start = dates[start_idx]
        test_end = dates[end_idx - 1]

        train_df = df[df["target_session_date"] <= train_end]
        test_df = df[(df["target_session_date"] >= test_start) & (df["target_session_date"] <= test_end)]

        if len(train_df) < 100 or len(test_df) < 10:
            start_idx += step_days
            continue

        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)

        common_cols = [c for c in X_train.columns if c in X_test.columns]
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        step_acc = accuracy_score(y_test, y_pred)
        step_auc = roc_auc_score(y_test, y_prob) if len(y_test.unique()) > 1 else 0.5

        step_results.append({
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "accuracy": step_acc,
            "auc_roc": step_auc,
        })

        for i, (_, row) in enumerate(test_df.iterrows()):
            all_predictions.append({
                "symbol": row["symbol"],
                "date": row["target_session_date"],
                "actual": int(y_test.iloc[i]),
                "predicted": int(y_pred[i]),
                "probability_up": float(y_prob[i]),
            })

        logger.info(
            f"  Step {len(step_results)}: "
            f"train<={train_end}, test={test_start}→{test_end}, "
            f"acc={step_acc:.4f}, auc={step_auc:.4f}"
        )

        start_idx += step_days

    if not step_results:
        return {"error": "no valid steps"}

    preds_df = pd.DataFrame(all_predictions)
    agg_metrics = _compute_aggregate_metrics(preds_df, step_results)

    logger.info(f"Walk-forward backtest complete:")
    logger.info(f"  Steps: {len(step_results)}")
    logger.info(f"  Overall accuracy: {agg_metrics['overall_accuracy']:.4f}")
    logger.info(f"  Overall AUC: {agg_metrics['overall_auc']:.4f}")
    logger.info(f"  Avg step accuracy: {agg_metrics['avg_step_accuracy']:.4f}")

    if agg_metrics["overall_accuracy"] > 0.65:
        logger.warning(
            f"LEAKAGE ALERT: Backtest accuracy {agg_metrics['overall_accuracy']:.4f} > 0.65"
        )

    return {
        "step_results": step_results,
        "predictions": preds_df,
        "aggregate_metrics": agg_metrics,
    }


def _compute_aggregate_metrics(preds_df: pd.DataFrame, step_results: list[dict]) -> dict:
    """Compute aggregate backtest metrics."""
    overall_acc = accuracy_score(preds_df["actual"], preds_df["predicted"])
    overall_auc = roc_auc_score(preds_df["actual"], preds_df["probability_up"]) \
        if preds_df["actual"].nunique() > 1 else 0.5

    step_accs = [s["accuracy"] for s in step_results]

    confident = preds_df[preds_df["probability_up"].apply(lambda p: max(p, 1 - p)) >= 0.6]
    confident_acc = accuracy_score(confident["actual"], confident["predicted"]) if len(confident) > 0 else None

    return {
        "overall_accuracy": overall_acc,
        "overall_auc": overall_auc,
        "avg_step_accuracy": np.mean(step_accs),
        "std_step_accuracy": np.std(step_accs),
        "min_step_accuracy": np.min(step_accs),
        "max_step_accuracy": np.max(step_accs),
        "n_steps": len(step_results),
        "n_predictions": len(preds_df),
        "confident_accuracy": confident_acc,
        "confident_n": len(confident),
        "pct_up_predicted": (preds_df["predicted"] == 1).mean(),
        "pct_up_actual": (preds_df["actual"] == 1).mean(),
    }
