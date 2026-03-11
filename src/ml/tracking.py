"""MLflow tracking standards (ML-301).

Provides standardized helpers for experiment logging with required metadata:
git hash, dataset version, pipeline version, training window tags.
"""

import logging
import os
import subprocess

import mlflow

from src.config import settings

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1.0.0"


def configure_mlflow():
    """Configure MLflow tracking URI and credentials."""
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
        logger.info(f"MLflow configured: {settings.mlflow_tracking_uri}")
    else:
        logger.warning("No MLflow tracking URI set, using local tracking")


def get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def set_standard_tags(
    dataset_version: str | None = None,
    training_window_start: str | None = None,
    training_window_end: str | None = None,
    model_type: str = "xgboost",
):
    """Set mandatory MLflow tags per ENG-SPEC 10.1."""
    mlflow.set_tag("git_commit", get_git_hash())
    mlflow.set_tag("pipeline_version", PIPELINE_VERSION)
    mlflow.set_tag("model_type", model_type)

    if dataset_version:
        mlflow.set_tag("dataset_version", dataset_version)
    if training_window_start:
        mlflow.set_tag("training_window_start", training_window_start)
    if training_window_end:
        mlflow.set_tag("training_window_end", training_window_end)


def log_training_run(
    experiment_name: str,
    run_name: str | None,
    params: dict,
    metrics: dict,
    model,
    feature_columns: list[str],
    dataset_version: str | None = None,
    training_window: tuple[str, str] | None = None,
    importance: dict | None = None,
    calibration_result: dict | None = None,
) -> str:
    """Execute a full MLflow-standardized training run.

    Returns the MLflow run_id.
    """
    configure_mlflow()

    try:
        mlflow.set_experiment(experiment_name)
    except Exception:
        logger.warning("MLflow experiment setup failed, using default")

    with mlflow.start_run(run_name=run_name) as run:
        set_standard_tags(
            dataset_version=dataset_version,
            training_window_start=training_window[0] if training_window else None,
            training_window_end=training_window[1] if training_window else None,
        )

        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("feature_columns", str(feature_columns))

        mlflow.log_metrics(metrics)

        if importance:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in top_features:
                mlflow.log_metric(f"importance_{feat}", float(imp))

        mlflow.xgboost.log_model(model, "model")

        if calibration_result:
            cal_metrics = calibration_result.get("metrics", {})
            mlflow.log_metrics({f"cal_{k}": v for k, v in cal_metrics.items()})
            if "calibration_curve" in calibration_result:
                mlflow.log_dict(calibration_result["calibration_curve"], "calibration_curve.json")

        run_id = run.info.run_id
        logger.info(f"MLflow run logged: {run_id} (experiment: {experiment_name})")
        return run_id
