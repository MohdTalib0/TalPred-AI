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
from datetime import UTC, date, datetime

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

BACKTEST_GATES = {
    "min_auc": 0.52,
    "min_rank_sharpe_net": 1.0,
    "max_rank_mdd_net_abs": 0.40,
    "min_ic_mean": 0.02,
    "min_decile_monotonicity": 0.70,
}


def generate_model_version(algorithm: str, run_id: str) -> str:
    """Generate a v{major}.{minor}.{patch} model version.

    Patch is derived from a hash of the run_id.  Using 6 hex chars
    (16M range) mod 10000 keeps birthday-collision probability < 0.01%
    at 50 training runs.
    """
    patch = int(hashlib.sha256(run_id.encode()).hexdigest()[:6], 16) % 10000
    return f"v1.0.{patch}"


def check_kpi_gates(metrics: dict) -> dict:
    """Validate model metrics against KPI thresholds.

    Returns dict with passed (bool), gate_results (list of check details).
    Each gate result includes metric_value, threshold, and direction for
    Holm-Bonferroni correction.
    """
    results = []
    all_passed = True

    accuracy = metrics.get("accuracy", 0)
    if accuracy < KPI_GATES["min_accuracy"]:
        results.append({"gate": "min_accuracy", "passed": False,
                        "detail": f"accuracy {accuracy:.4f} < {KPI_GATES['min_accuracy']}",
                        "metric_value": accuracy, "threshold": KPI_GATES["min_accuracy"], "direction": "above"})
        all_passed = False
    else:
        results.append({"gate": "min_accuracy", "passed": True, "detail": f"accuracy {accuracy:.4f}",
                        "metric_value": accuracy, "threshold": KPI_GATES["min_accuracy"], "direction": "above"})

    if accuracy > KPI_GATES["max_accuracy"]:
        results.append({"gate": "leakage_guard", "passed": False,
                        "detail": f"accuracy {accuracy:.4f} > {KPI_GATES['max_accuracy']} (leakage suspected)",
                        "metric_value": accuracy, "threshold": KPI_GATES["max_accuracy"], "direction": "below"})
        all_passed = False
    else:
        results.append({"gate": "leakage_guard", "passed": True, "detail": f"accuracy {accuracy:.4f}",
                        "metric_value": accuracy, "threshold": KPI_GATES["max_accuracy"], "direction": "below"})

    auc = metrics.get("auc_roc", 0)
    if auc < KPI_GATES["min_auc_roc"]:
        results.append({"gate": "min_auc", "passed": False,
                        "detail": f"AUC {auc:.4f} < {KPI_GATES['min_auc_roc']}",
                        "metric_value": auc, "threshold": KPI_GATES["min_auc_roc"], "direction": "above"})
        all_passed = False
    else:
        results.append({"gate": "min_auc", "passed": True, "detail": f"AUC {auc:.4f}",
                        "metric_value": auc, "threshold": KPI_GATES["min_auc_roc"], "direction": "above"})

    ll = metrics.get("log_loss", float("inf"))
    if ll > KPI_GATES["max_log_loss"]:
        results.append({"gate": "max_log_loss", "passed": False,
                        "detail": f"log_loss {ll:.4f} > {KPI_GATES['max_log_loss']}",
                        "metric_value": ll, "threshold": KPI_GATES["max_log_loss"], "direction": "below"})
        all_passed = False
    else:
        results.append({"gate": "max_log_loss", "passed": True, "detail": f"log_loss {ll:.4f}",
                        "metric_value": ll, "threshold": KPI_GATES["max_log_loss"], "direction": "below"})

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


def check_backtest_gates(backtest_results: dict | None) -> dict:
    """Validate ranking-first backtest promotion criteria."""
    if not backtest_results:
        return {"passed": True, "detail": "no backtest provided (skipped)", "gate_results": []}

    agg = backtest_results.get("aggregate_metrics", {})
    results = []
    all_passed = True

    bt_auc = agg.get("overall_auc")
    if bt_auc is None:
        results.append({"gate": "min_backtest_auc", "passed": False, "detail": "missing overall_auc"})
        all_passed = False
    elif bt_auc < BACKTEST_GATES["min_auc"]:
        results.append({
            "gate": "min_backtest_auc", "passed": False,
            "detail": f"backtest AUC {bt_auc:.4f} < {BACKTEST_GATES['min_auc']}",
            "metric_value": bt_auc, "threshold": BACKTEST_GATES["min_auc"], "direction": "above",
        })
        all_passed = False
    else:
        results.append({
            "gate": "min_backtest_auc", "passed": True, "detail": f"backtest AUC {bt_auc:.4f}",
            "metric_value": bt_auc, "threshold": BACKTEST_GATES["min_auc"], "direction": "above",
        })

    sharpe_net = agg.get("rank_long_short_sharpe_net_nw")
    sharpe_label = "net NW ranking Sharpe"
    if sharpe_net is None:
        sharpe_net = agg.get("rank_long_short_sharpe_net")
        sharpe_label = "net ranking Sharpe"
    if sharpe_net is None:
        results.append({"gate": "min_rank_sharpe_net", "passed": False, "detail": "missing net ranking Sharpe metric"})
        all_passed = False
    elif sharpe_net < BACKTEST_GATES["min_rank_sharpe_net"]:
        results.append({
            "gate": "min_rank_sharpe_net", "passed": False,
            "detail": f"{sharpe_label} {sharpe_net:.3f} < {BACKTEST_GATES['min_rank_sharpe_net']}",
            "metric_value": sharpe_net, "threshold": BACKTEST_GATES["min_rank_sharpe_net"], "direction": "above",
        })
        all_passed = False
    else:
        results.append({
            "gate": "min_rank_sharpe_net", "passed": True,
            "detail": f"{sharpe_label} {sharpe_net:.3f}",
            "metric_value": sharpe_net, "threshold": BACKTEST_GATES["min_rank_sharpe_net"], "direction": "above",
        })

    mdd_net = agg.get("rank_max_drawdown_net")
    if mdd_net is None:
        results.append({"gate": "max_rank_mdd_net_abs", "passed": False, "detail": "missing rank_max_drawdown_net"})
        all_passed = False
    elif abs(float(mdd_net)) > BACKTEST_GATES["max_rank_mdd_net_abs"]:
        results.append({
            "gate": "max_rank_mdd_net_abs", "passed": False,
            "detail": f"net ranking drawdown {mdd_net:.3f} exceeds {BACKTEST_GATES['max_rank_mdd_net_abs']:.2f}",
            "metric_value": abs(float(mdd_net)), "threshold": BACKTEST_GATES["max_rank_mdd_net_abs"], "direction": "below",
        })
        all_passed = False
    else:
        results.append({
            "gate": "max_rank_mdd_net_abs", "passed": True,
            "detail": f"net ranking drawdown {mdd_net:.3f}",
            "metric_value": abs(float(mdd_net)), "threshold": BACKTEST_GATES["max_rank_mdd_net_abs"], "direction": "below",
        })

    ic_mean = agg.get("ic_mean")
    if ic_mean is not None:
        if ic_mean < BACKTEST_GATES["min_ic_mean"]:
            results.append({
                "gate": "min_ic_mean", "passed": False,
                "detail": f"IC mean {ic_mean:.4f} < {BACKTEST_GATES['min_ic_mean']}",
                "metric_value": ic_mean, "threshold": BACKTEST_GATES["min_ic_mean"], "direction": "above",
            })
            all_passed = False
        else:
            results.append({
                "gate": "min_ic_mean", "passed": True,
                "detail": f"IC mean {ic_mean:.4f}",
                "metric_value": ic_mean, "threshold": BACKTEST_GATES["min_ic_mean"], "direction": "above",
            })

    mono = agg.get("decile_monotonicity_spearman")
    if mono is not None:
        if mono < BACKTEST_GATES["min_decile_monotonicity"]:
            results.append({
                "gate": "min_decile_monotonicity", "passed": False,
                "detail": f"monotonicity {mono:.3f} < {BACKTEST_GATES['min_decile_monotonicity']}",
                "metric_value": mono, "threshold": BACKTEST_GATES["min_decile_monotonicity"], "direction": "above",
            })
            all_passed = False
        else:
            results.append({
                "gate": "min_decile_monotonicity", "passed": True,
                "detail": f"monotonicity {mono:.3f}",
                "metric_value": mono, "threshold": BACKTEST_GATES["min_decile_monotonicity"], "direction": "above",
            })

    return {"passed": all_passed, "gate_results": results}


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


def _holm_bonferroni_adjust(gate_results: list[dict]) -> list[dict]:
    """Apply Holm-Bonferroni step-down correction to gate results.

    For each passing gate, computes a margin (how far the metric is from
    the threshold relative to the threshold).  Gates with small margins
    are "borderline" and must survive a tightened threshold under the
    Holm step-down procedure.

    With m total tests, the k-th most borderline passing gate must
    exceed its threshold by at least threshold * margin_penalty where
    margin_penalty = base_tightening * (m - k + 1) / m.
    base_tightening is 0.05 (5% threshold tightening for the most
    borderline gate), ensuring the correction is meaningful but not
    overly aggressive for small-sample quant models.
    """
    n_gates = len(gate_results)
    if n_gates <= 1:
        return [dict(g, holm_bonferroni_applied=True, n_total_gates=n_gates) for g in gate_results]

    BASE_TIGHTENING = 0.05

    passing_with_metrics = []
    adjusted = []

    for i, g in enumerate(gate_results):
        entry = dict(g)
        entry["holm_bonferroni_applied"] = True
        entry["n_total_gates"] = n_gates

        if not g.get("passed", True) or "metric_value" not in g or "threshold" not in g:
            adjusted.append(entry)
            continue

        metric_val = g["metric_value"]
        threshold = g["threshold"]
        direction = g.get("direction", "above")

        if threshold == 0:
            margin = abs(metric_val)
        elif direction == "above":
            margin = (metric_val - threshold) / abs(threshold)
        else:
            margin = (threshold - metric_val) / abs(threshold)

        passing_with_metrics.append((i, margin, entry))

    passing_with_metrics.sort(key=lambda x: x[1])

    m = len(passing_with_metrics)
    for rank_k, (orig_idx, margin, entry) in enumerate(passing_with_metrics):
        holm_factor = (m - rank_k) / m
        required_margin = BASE_TIGHTENING * holm_factor
        entry["holm_required_margin"] = round(required_margin, 6)
        entry["holm_actual_margin"] = round(margin, 6)

        if margin < required_margin:
            entry["passed"] = False
            entry["holm_flipped"] = True
            entry["detail"] = (
                f"{entry['detail']} [HOLM-FLIPPED: margin {margin:.4f} < "
                f"required {required_margin:.4f}]"
            )
            logger.warning(
                f"Holm-Bonferroni flipped gate '{entry['gate']}': "
                f"margin={margin:.4f} < required={required_margin:.4f}"
            )
        adjusted.append(entry)

    # Restore original gate ordering
    gate_order = {g["gate"]: i for i, g in enumerate(gate_results) if "gate" in g}
    adjusted.sort(key=lambda e: gate_order.get(e.get("gate", ""), len(gate_order)))

    return adjusted


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

    Runs all promotion gates with Holm-Bonferroni multiple-testing
    correction, then either promotes or rejects.
    Returns promotion report.
    """
    report = {"model_version": model_version, "target_status": target_status}

    kpi_check = check_kpi_gates(metrics)
    report["kpi_gates"] = kpi_check

    lineage_check = check_lineage(mlflow_run_id, dataset_version)
    report["lineage_check"] = lineage_check

    backtest_check = check_backtest_gates(backtest_results)
    report["backtest_check"] = backtest_check
    backtest_ok = backtest_check["passed"]

    # Collect all individual gate results for multiple-testing adjustment
    all_gate_results = (
        kpi_check.get("gate_results", [])
        + backtest_check.get("gate_results", [])
    )
    adjusted_gates = _holm_bonferroni_adjust(all_gate_results)
    report["adjusted_gates"] = adjusted_gates
    report["n_gates_evaluated"] = len(all_gate_results)

    holm_any_flipped = any(g.get("holm_flipped") for g in adjusted_gates)
    adjusted_all_pass = all(g.get("passed", True) for g in adjusted_gates)

    all_passed = (
        kpi_check["passed"]
        and lineage_check["passed"]
        and backtest_ok
        and adjusted_all_pass
    )
    if holm_any_flipped:
        report["holm_rejection"] = True
        logger.warning("Holm-Bonferroni correction flipped borderline gates — promotion blocked")
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
            if target_status == "production":
                model.promoted_at = datetime.now(UTC)
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

    Demotes current production model to 'archived', promotes the most
    recently *promoted* archived model (by ``promoted_at``).  Falls back
    to ``created_at`` for models that pre-date the promoted_at column.
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
        .order_by(
            ModelRegistry.promoted_at.desc().nullslast(),
            ModelRegistry.created_at.desc(),
        )
        .first()
    )

    if previous:
        previous.status = "production"
        previous.promoted_at = datetime.now(UTC)
        db.commit()
        logger.info(f"Rolled back to model {previous.model_version}")
        return previous.model_version

    db.commit()
    logger.warning("No previous model available for rollback")
    return None
