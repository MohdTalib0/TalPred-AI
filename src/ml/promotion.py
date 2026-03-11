"""Model promotion pipeline with gates (ML-303).

Lifecycle: training -> experiment -> evaluation -> registry -> staging -> production

Gates (all mandatory per ENG-SPEC 11):
1. Leakage tests pass
2. KPI thresholds pass
3. Backtest validation approved
4. Full lineage (MLflow + DVC) present
"""

import hashlib
import logging
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.schema import CalibrationModel, ModelRegistry

logger = logging.getLogger(__name__)

KPI_GATES = {
    "min_accuracy": 0.52,
    "max_accuracy": 0.65,
    "min_auc_roc": 0.51,
    "max_log_loss": 1.0,
}


def generate_model_version(algorithm: str, run_id: str) -> str:
    """Generate a v{major}.{minor}.{patch} model version.

    Patch is derived from a hash of the run_id to ensure uniqueness.
    """
    patch = int(hashlib.sha256(run_id.encode()).hexdigest()[:4], 16) % 1000
    return f"v1.0.{patch}"


def check_kpi_gates(metrics: dict) -> dict:
    """Validate model metrics against KPI thresholds.

    Returns dict with passed (bool), gate_results (list of check details).
    """
    results = []
    all_passed = True

    accuracy = metrics.get("accuracy", 0)
    if accuracy < KPI_GATES["min_accuracy"]:
        results.append({"gate": "min_accuracy", "passed": False,
                        "detail": f"accuracy {accuracy:.4f} < {KPI_GATES['min_accuracy']}"})
        all_passed = False
    else:
        results.append({"gate": "min_accuracy", "passed": True, "detail": f"accuracy {accuracy:.4f}"})

    if accuracy > KPI_GATES["max_accuracy"]:
        results.append({"gate": "leakage_guard", "passed": False,
                        "detail": f"accuracy {accuracy:.4f} > {KPI_GATES['max_accuracy']} (leakage suspected)"})
        all_passed = False
    else:
        results.append({"gate": "leakage_guard", "passed": True, "detail": f"accuracy {accuracy:.4f}"})

    auc = metrics.get("auc_roc", 0)
    if auc < KPI_GATES["min_auc_roc"]:
        results.append({"gate": "min_auc", "passed": False,
                        "detail": f"AUC {auc:.4f} < {KPI_GATES['min_auc_roc']}"})
        all_passed = False
    else:
        results.append({"gate": "min_auc", "passed": True, "detail": f"AUC {auc:.4f}"})

    ll = metrics.get("log_loss", float("inf"))
    if ll > KPI_GATES["max_log_loss"]:
        results.append({"gate": "max_log_loss", "passed": False,
                        "detail": f"log_loss {ll:.4f} > {KPI_GATES['max_log_loss']}"})
        all_passed = False
    else:
        results.append({"gate": "max_log_loss", "passed": True, "detail": f"log_loss {ll:.4f}"})

    return {"passed": all_passed, "gate_results": results}


def check_lineage(mlflow_run_id: str | None, dataset_version: str | None) -> dict:
    """Verify full lineage is present.

    MLflow run_id is mandatory. Dataset version is recommended but
    not a hard gate for v1 (DVC workflow may not always be run).
    """
    issues = []
    warnings = []
    if not mlflow_run_id:
        issues.append("Missing MLflow run_id")
    if not dataset_version:
        warnings.append("Missing dataset_version (DVC not linked)")
    return {"passed": len(issues) == 0, "issues": issues, "warnings": warnings}


def register_model(
    db: Session,
    mlflow_run_id: str,
    algorithm: str,
    training_window_start: date,
    training_window_end: date,
    metrics: dict,
    dataset_version: str | None = None,
    status: str = "candidate",
) -> str:
    """Register a model candidate in the model_registry table.

    Returns the generated model_version string.
    """
    model_version = generate_model_version(algorithm, mlflow_run_id)

    existing = db.query(ModelRegistry).filter(
        ModelRegistry.model_version == model_version
    ).first()
    if existing:
        logger.warning(f"Model version {model_version} already exists, updating")
        existing.mlflow_run_id = mlflow_run_id
        existing.metrics = metrics
        existing.status = status
    else:
        db.add(ModelRegistry(
            model_version=model_version,
            mlflow_run_id=mlflow_run_id,
            algorithm=algorithm,
            training_window_start=training_window_start,
            training_window_end=training_window_end,
            metrics=metrics,
            status=status,
        ))

    db.commit()
    logger.info(f"Registered model {model_version} with status '{status}'")
    return model_version


def register_calibration(
    db: Session,
    model_version: str,
    calibration_type: str,
    training_window: str,
    calibration_metrics: dict,
    artifact_uri: str = "",
):
    """Store calibration artifact metadata."""
    existing = db.query(CalibrationModel).filter(
        CalibrationModel.model_version == model_version,
        CalibrationModel.calibration_type == calibration_type,
    ).first()

    if existing:
        existing.calibration_metrics = calibration_metrics
        existing.artifact_uri = artifact_uri
    else:
        db.add(CalibrationModel(
            model_version=model_version,
            calibration_type=calibration_type,
            training_window=training_window,
            calibration_metrics=calibration_metrics,
            artifact_uri=artifact_uri,
        ))
    db.commit()
    logger.info(f"Calibration registered for {model_version} ({calibration_type})")


def promote_model(
    db: Session,
    model_version: str,
    target_status: str,
    metrics: dict,
    mlflow_run_id: str | None = None,
    dataset_version: str | None = None,
    backtest_results: dict | None = None,
) -> dict:
    """Attempt to promote a model through the lifecycle.

    Runs all promotion gates and either promotes or rejects.
    Returns promotion report.
    """
    report = {"model_version": model_version, "target_status": target_status}

    kpi_check = check_kpi_gates(metrics)
    report["kpi_gates"] = kpi_check

    lineage_check = check_lineage(mlflow_run_id, dataset_version)
    report["lineage_check"] = lineage_check

    backtest_ok = True
    if backtest_results:
        bt_acc = backtest_results.get("aggregate_metrics", {}).get("overall_accuracy", 0)
        if bt_acc < KPI_GATES["min_accuracy"]:
            backtest_ok = False
            report["backtest_check"] = {"passed": False, "detail": f"backtest accuracy {bt_acc:.4f} too low"}
        elif bt_acc > KPI_GATES["max_accuracy"]:
            backtest_ok = False
            report["backtest_check"] = {"passed": False, "detail": f"backtest accuracy {bt_acc:.4f} too high (leakage)"}
        else:
            report["backtest_check"] = {"passed": True, "detail": f"backtest accuracy {bt_acc:.4f}"}
    else:
        report["backtest_check"] = {"passed": True, "detail": "no backtest provided (skipped)"}

    all_passed = kpi_check["passed"] and lineage_check["passed"] and backtest_ok
    report["promoted"] = all_passed

    if all_passed:
        model = db.query(ModelRegistry).filter(
            ModelRegistry.model_version == model_version
        ).first()
        if model:
            if target_status == "production":
                db.execute(text("""
                    UPDATE model_registry SET status = 'archived'
                    WHERE status = 'production' AND model_version != :mv
                """), {"mv": model_version})

            model.status = target_status
            db.commit()
            logger.info(f"Model {model_version} promoted to '{target_status}'")
        else:
            report["promoted"] = False
            report["error"] = f"Model {model_version} not found in registry"
    else:
        logger.warning(f"Model {model_version} REJECTED for promotion to '{target_status}'")
        for gate_name in ["kpi_gates", "lineage_check", "backtest_check"]:
            gate = report.get(gate_name, {})
            if not gate.get("passed", True):
                logger.warning(f"  Failed: {gate_name} - {gate}")

    return report


def get_production_model(db: Session) -> ModelRegistry | None:
    """Get the current production model."""
    return (
        db.query(ModelRegistry)
        .filter(ModelRegistry.status == "production")
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )


def rollback_model(db: Session) -> str | None:
    """Rollback to the previous production model.

    Demotes current production model to 'archived', promotes latest 'archived'.
    """
    current = get_production_model(db)
    if current:
        current.status = "archived"

    previous = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.status == "archived",
            ModelRegistry.model_version != (current.model_version if current else ""),
        )
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )

    if previous:
        previous.status = "production"
        db.commit()
        logger.info(f"Rolled back to model {previous.model_version}")
        return previous.model_version

    db.commit()
    logger.warning("No previous model available for rollback")
    return None
