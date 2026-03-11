"""Probability calibration (ML-207).

Calibrates XGBoost output probabilities using Platt scaling (logistic) or
isotonic regression so that confidence scores are reliable for trading thresholds.
"""

import logging
import pickle

import mlflow
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.models.trainer import prepare_features

logger = logging.getLogger(__name__)


def calibrate_model(
    model: xgb.XGBClassifier,
    df_calibration: "pd.DataFrame",
    method: str = "isotonic",
) -> dict:
    """Calibrate a trained model using held-out calibration data.

    Args:
        model: Trained XGBClassifier
        df_calibration: Calibration dataset (separate from train/val)
        method: "sigmoid" (Platt) or "isotonic"

    Returns dict with calibrated model, metrics, and calibration curve data.
    """
    X_cal, y_cal = prepare_features(df_calibration)

    raw_probs = model.predict_proba(X_cal)[:, 1]
    raw_brier = brier_score_loss(y_cal, raw_probs)

    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)

    cal_probs = calibrated.predict_proba(X_cal)[:, 1]
    cal_brier = brier_score_loss(y_cal, cal_probs)

    prob_true_raw, prob_pred_raw = calibration_curve(y_cal, raw_probs, n_bins=10, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_cal, cal_probs, n_bins=10, strategy="uniform")

    raw_ece = _expected_calibration_error(prob_true_raw, prob_pred_raw)
    cal_ece = _expected_calibration_error(prob_true_cal, prob_pred_cal)

    logger.info(f"Calibration ({method}):")
    logger.info(f"  Raw  - Brier: {raw_brier:.4f}, ECE: {raw_ece:.4f}")
    logger.info(f"  Cal  - Brier: {cal_brier:.4f}, ECE: {cal_ece:.4f}")
    logger.info(f"  Improvement: Brier {(raw_brier - cal_brier) / raw_brier * 100:.1f}%")

    return {
        "calibrated_model": calibrated,
        "method": method,
        "metrics": {
            "raw_brier": raw_brier,
            "calibrated_brier": cal_brier,
            "raw_ece": raw_ece,
            "calibrated_ece": cal_ece,
            "brier_improvement_pct": (raw_brier - cal_brier) / raw_brier * 100,
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
