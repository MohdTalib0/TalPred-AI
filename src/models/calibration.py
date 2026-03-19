"""Probability calibration (ML-207).

Calibrates XGBoost output probabilities using Platt scaling (logistic) or
isotonic regression so that confidence scores are reliable for trading thresholds.
"""

import logging

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from src.models.trainer import prepare_features

logger = logging.getLogger(__name__)


def calibrate_model(
    model: xgb.XGBClassifier,
    df_calibration: pd.DataFrame,
    method: str = "isotonic",
    train_medians: pd.Series | None = None,
    feature_profile: str = "all_features",
) -> dict:
    """Calibrate a trained model using held-out calibration data.

    Splits the calibration set into a fit portion (70%) for fitting the
    calibrator and an eval portion (30%) for unbiased metric reporting.

    Args:
        model: Trained XGBClassifier
        df_calibration: Calibration dataset (separate from train/val)
        method: "sigmoid" (Platt) or "isotonic"
        train_medians: Medians from training set for NaN filling

    Returns dict with calibrated model, metrics, and calibration curve data.
    """
    split_idx = int(len(df_calibration) * 0.7)
    df_fit = df_calibration.iloc[:split_idx]
    df_eval = df_calibration.iloc[split_idx:]

    X_fit, y_fit, medians = prepare_features(df_fit, fill_medians=train_medians, feature_profile=feature_profile)
    X_eval, y_eval, _ = prepare_features(df_eval, fill_medians=medians, feature_profile=feature_profile)

    model_features = model.get_booster().feature_names
    for feat_set in (X_fit, X_eval):
        for col in model_features:
            if col not in feat_set.columns:
                feat_set[col] = 0
    X_fit = X_fit[model_features]
    X_eval = X_eval[model_features]

    raw_probs = model.predict_proba(X_eval)[:, 1]
    raw_brier = brier_score_loss(y_eval, raw_probs)

    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(X_fit, y_fit)

    cal_probs = calibrated.predict_proba(X_eval)[:, 1]
    cal_brier = brier_score_loss(y_eval, cal_probs)

    prob_true_raw, prob_pred_raw = calibration_curve(y_eval, raw_probs, n_bins=10, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_eval, cal_probs, n_bins=10, strategy="uniform")

    raw_ece = _expected_calibration_error(prob_true_raw, prob_pred_raw)
    cal_ece = _expected_calibration_error(prob_true_cal, prob_pred_cal)

    logger.info(f"Calibration ({method}), eval on {len(df_eval)} held-out rows:")
    logger.info(f"  Raw  - Brier: {raw_brier:.4f}, ECE: {raw_ece:.4f}")
    logger.info(f"  Cal  - Brier: {cal_brier:.4f}, ECE: {cal_ece:.4f}")
    improvement = (raw_brier - cal_brier) / raw_brier * 100 if raw_brier > 0 else 0
    logger.info(f"  Improvement: Brier {improvement:.1f}%")

    return {
        "calibrated_model": calibrated,
        "method": method,
        "metrics": {
            "raw_brier": raw_brier,
            "calibrated_brier": cal_brier,
            "raw_ece": raw_ece,
            "calibrated_ece": cal_ece,
            "brier_improvement_pct": improvement,
        },
        "calibration_curve": {
            "raw": {"prob_true": prob_true_raw.tolist(), "prob_pred": prob_pred_raw.tolist()},
            "calibrated": {"prob_true": prob_true_cal.tolist(), "prob_pred": prob_pred_cal.tolist()},
        },
    }


def _expected_calibration_error(prob_true: np.ndarray, prob_pred: np.ndarray) -> float:
    """Compute Expected Calibration Error."""
    return np.mean(np.abs(prob_true - prob_pred))


def log_calibration_to_mlflow(calibration_result: dict, run_id: str | None = None):
    """Log calibration artifacts and metrics to MLflow."""
    metrics = calibration_result["metrics"]

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                f"cal_{k}": v for k, v in metrics.items()
            })
            mlflow.log_dict(calibration_result["calibration_curve"], "calibration_curve.json")
    else:
        mlflow.log_metrics({f"cal_{k}": v for k, v in metrics.items()})
        mlflow.log_dict(calibration_result["calibration_curve"], "calibration_curve.json")

    logger.info("Calibration results logged to MLflow")
