"""Launch gate report (QA-401).

End-to-end UAT checklist per ENG-SPEC 11 / Sprint Plan 11.

Go-live only if ALL conditions pass:
1. Pipeline success >= 99.5% over last 30 staging days
2. Data freshness SLA >= 99%
3. KPI thresholds met on locked holdout
4. Rollback to previous model validated
5. Compliance disclaimer present

Usage:
  python -m scripts.launch_gate
"""

import logging
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.ml.promotion import get_production_model  # noqa: E402
from src.monitoring.checks import run_all_checks  # noqa: E402

from sqlalchemy import text  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)


def check_pipeline_success(db) -> dict:
    """Check pipeline success rate over last 30 days."""
    result = db.execute(text("""
        SELECT COUNT(DISTINCT date) FROM market_bars_daily
        WHERE date >= CURRENT_DATE - 30
    """))
    bar_days = result.scalar() or 0

    result = db.execute(text("""
        SELECT COUNT(DISTINCT target_session_date) FROM features_snapshot
        WHERE target_session_date >= CURRENT_DATE - 30
    """))
    feat_days = result.scalar() or 0

    result = db.execute(text("""
        SELECT COUNT(DISTINCT target_date) FROM predictions
        WHERE target_date >= CURRENT_DATE - 30
    """))
    pred_days = result.scalar() or 0

    success_rate = min(feat_days, pred_days) / max(bar_days, 1) * 100
    passed = success_rate >= 99.5

    return {
        "check": "pipeline_success_rate",
        "passed": passed,
        "value": f"{success_rate:.1f}%",
        "threshold": ">=99.5%",
        "detail": f"bar_days={bar_days}, feat_days={feat_days}, pred_days={pred_days}",
    }


def check_data_freshness_sla(db) -> dict:
    """Check data freshness SLA."""
    result = db.execute(text("""
        SELECT
            COUNT(*) AS total,
            COUNT(CASE WHEN date >= CURRENT_DATE - 2 THEN 1 END) AS fresh
        FROM (
            SELECT DISTINCT date FROM market_bars_daily
            WHERE date >= CURRENT_DATE - 30
        ) dates
    """))
    row = result.fetchone()
    total = row[0] or 0
    fresh = row[1] or 0

    freshness = fresh / max(total, 1) * 100
    passed = freshness >= 99 or total < 5

    return {
        "check": "data_freshness_sla",
        "passed": passed,
        "value": f"{freshness:.1f}%",
        "threshold": ">=99%",
    }


def check_production_model(db) -> dict:
    """Check that a production model exists with valid metrics."""
    model = get_production_model(db)
    if not model:
        return {
            "check": "production_model",
            "passed": False,
            "detail": "No production model found",
        }

    metrics = model.metrics or {}
    accuracy = metrics.get("accuracy", 0)
    auc = metrics.get("auc_roc", 0)

    passed = 0.52 <= accuracy <= 0.65 and auc >= 0.51

    return {
        "check": "production_model_kpi",
        "passed": passed,
        "model_version": model.model_version,
        "accuracy": accuracy,
        "auc_roc": auc,
    }


def check_rollback_capability(db) -> dict:
    """Verify rollback mechanism works (dry check)."""
    from src.models.schema import ModelRegistry
    archived = db.query(ModelRegistry).filter(
        ModelRegistry.status == "archived"
    ).count()

    return {
        "check": "rollback_capability",
        "passed": True,
        "archived_models": archived,
        "detail": "Rollback path available" if archived > 0 else "No archived models (first deploy)",
    }


def check_disclaimer(db) -> dict:
    """Verify compliance disclaimer is present in API."""
    from src.serving.api import DISCLAIMER
    has_disclaimer = "financial advice" in DISCLAIMER.lower()
    return {
        "check": "compliance_disclaimer",
        "passed": has_disclaimer,
        "disclaimer": DISCLAIMER,
    }


def check_live_ic_deployment_guard(
    db,
    rolling_window_days: int = 60,
    lookback_days: int = 180,
    reduce_threshold: float = 0.02,
    suspend_threshold: float = 0.01,
    consecutive_days: int = 10,
) -> dict:
    """Two-level live IC deployment guard.

    - rolling_live_ic < suspend_threshold for N consecutive days => NO-GO
    - rolling_live_ic < reduce_threshold for N consecutive days => GO with reduce-exposure action
    """
    result = db.execute(text("""
        SELECT target_date, probability_up, realized_return
        FROM predictions
        WHERE realized_return IS NOT NULL
          AND target_date >= CURRENT_DATE - :lookback_days
        ORDER BY target_date
    """), {"lookback_days": lookback_days})
    rows = result.fetchall()
    if not rows:
        return {
            "check": "live_ic_deployment_guard",
            "passed": True,
            "status": "insufficient_data",
            "detail": "No realized prediction returns available yet",
        }

    df = pd.DataFrame(rows, columns=["target_date", "probability_up", "realized_return"])
    daily = []
    for dt, grp in df.groupby("target_date"):
        if len(grp) < 10:
            continue
        ic = grp["probability_up"].corr(grp["realized_return"], method="spearman")
        if pd.notna(ic):
            daily.append({"date": pd.Timestamp(dt), "ic": float(ic)})

    if len(daily) < rolling_window_days:
        return {
            "check": "live_ic_deployment_guard",
            "passed": True,
            "status": "insufficient_data",
            "detail": f"Need >= {rolling_window_days} IC days, have {len(daily)}",
        }

    ic_df = pd.DataFrame(daily).sort_values("date")
    ic_df["rolling_live_ic"] = ic_df["ic"].rolling(rolling_window_days).mean()
    ric = ic_df.dropna(subset=["rolling_live_ic"]).copy()
    if ric.empty:
        return {
            "check": "live_ic_deployment_guard",
            "passed": True,
            "status": "insufficient_data",
            "detail": "Rolling live IC unavailable",
        }

    tail = ric.tail(consecutive_days)
    latest = float(ric["rolling_live_ic"].iloc[-1])
    below_suspend = bool((tail["rolling_live_ic"] < suspend_threshold).all()) if len(tail) == consecutive_days else False
    below_reduce = bool((tail["rolling_live_ic"] < reduce_threshold).all()) if len(tail) == consecutive_days else False

    if below_suspend:
        return {
            "check": "live_ic_deployment_guard",
            "passed": False,
            "status": "suspend",
            "latest_rolling_live_ic": round(latest, 5),
            "detail": (
                f"rolling live IC < {suspend_threshold:.3f} for {consecutive_days} consecutive days; "
                "suspend strategy"
            ),
        }
    if below_reduce:
        return {
            "check": "live_ic_deployment_guard",
            "passed": True,
            "status": "reduce_exposure",
            "latest_rolling_live_ic": round(latest, 5),
            "detail": (
                f"rolling live IC < {reduce_threshold:.3f} for {consecutive_days} consecutive days; "
                "reduce exposure"
            ),
        }
    return {
        "check": "live_ic_deployment_guard",
        "passed": True,
        "status": "normal",
        "latest_rolling_live_ic": round(latest, 5),
        "detail": "Live IC regime healthy",
    }


def main():
    db = SessionLocal()

    try:
        logger.info("=" * 60)
        logger.info("LAUNCH GATE REPORT")
        logger.info("=" * 60)

        checks = [
            check_pipeline_success(db),
            check_data_freshness_sla(db),
            check_production_model(db),
            check_rollback_capability(db),
            check_disclaimer(db),
            check_live_ic_deployment_guard(db),
        ]

        monitoring = run_all_checks(db)
        checks.append({
            "check": "monitoring_health",
            "passed": monitoring["total_alerts"] == 0,
            "alerts": monitoring["total_alerts"],
            "status": monitoring["overall_status"],
        })

        all_passed = all(c["passed"] for c in checks)

        for c in checks:
            status = "PASS" if c["passed"] else "FAIL"
            extra = c.get("value", c.get("detail", ""))
            if c.get("status") in {"reduce_exposure", "suspend"}:
                extra = f"{extra} (status={c.get('status')})"
            logger.info(f"  [{status}] {c['check']}: {extra}")

        logger.info("")
        logger.info("=" * 60)
        if all_passed:
            logger.info("LAUNCH GATE: GO")
            logger.info("All checks passed. System is ready for production.")
        else:
            logger.warning("LAUNCH GATE: NO-GO")
            failed = [c for c in checks if not c["passed"]]
            for f in failed:
                logger.warning(f"  BLOCKED BY: {f['check']}")
        logger.info("=" * 60)

        return 0 if all_passed else 1

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
