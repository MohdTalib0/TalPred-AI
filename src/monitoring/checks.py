"""Production monitoring checks (OP-401 / OP-402).

Automated health checks per ENG-SPEC 15:
- Data quality: missing bars, ingestion failures
- Model drift: PSI-based feature drift detection
- Pipeline health: freshness, latency SLOs
- Alert thresholds enforced
"""

import logging
from datetime import UTC, date, datetime

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

ALERT_THRESHOLDS = {
    "missing_bars_pct_30d": 1.0,
    "ingestion_failures_7d": 3,
    "data_latency_hours": 6,
    "psi_warning": 0.15,
    "psi_critical": 0.25,
}


def run_all_checks(db: Session) -> dict:
    """Run full monitoring suite. Returns report with all check results."""
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "data_quality": check_data_quality(db),
        "data_freshness": check_data_freshness(db),
        "model_performance": check_model_performance(db),
        "signal_environment": check_signal_environment(db),
        "feature_drift": check_feature_drift(db),
        "pipeline_health": check_pipeline_health(db),
    }

    alerts = []
    for section, checks in report.items():
        if isinstance(checks, dict) and "alerts" in checks:
            alerts.extend(checks["alerts"])

    report["total_alerts"] = len(alerts)
    report["alert_details"] = alerts
    report["overall_status"] = "healthy" if len(alerts) == 0 else "degraded"

    logger.info(
        f"Monitoring: {report['overall_status']} "
        f"({report['total_alerts']} alerts)"
    )
    return report


def check_data_quality(db: Session) -> dict:
    """Check for missing market bars and data gaps."""
    alerts = []

    result = db.execute(text("""
        SELECT
            COUNT(DISTINCT date) AS total_days,
            COUNT(DISTINCT symbol) AS total_symbols,
            COUNT(*) AS total_bars
        FROM market_bars_daily
        WHERE date >= CURRENT_DATE - 30
    """))
    row = result.fetchone()
    total_days = row[0] or 0
    total_symbols = row[1] or 0
    total_bars = row[2] or 0

    result = db.execute(text("""
        SELECT COUNT(*) FROM symbols WHERE is_active = true
    """))
    active_symbols = result.scalar() or 0

    expected_bars = total_days * active_symbols
    if expected_bars > 0:
        missing_pct = ((expected_bars - total_bars) / expected_bars) * 100
        if missing_pct > ALERT_THRESHOLDS["missing_bars_pct_30d"]:
            alerts.append({
                "level": "critical",
                "check": "missing_bars",
                "detail": f"{missing_pct:.2f}% bars missing in last 30 days "
                          f"(threshold: {ALERT_THRESHOLDS['missing_bars_pct_30d']}%)",
            })
    else:
        missing_pct = 0

    result = db.execute(text("""
        SELECT COUNT(*) FROM quarantine
        WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    """))
    quarantine_count = result.scalar() or 0
    if quarantine_count > ALERT_THRESHOLDS["ingestion_failures_7d"]:
        alerts.append({
            "level": "warning",
            "check": "ingestion_failures",
            "detail": f"{quarantine_count} quarantine records in 7 days "
                      f"(threshold: {ALERT_THRESHOLDS['ingestion_failures_7d']})",
        })

    return {
        "total_bars_30d": total_bars,
        "active_symbols": active_symbols,
        "missing_pct": round(missing_pct, 2),
        "quarantine_7d": quarantine_count,
        "alerts": alerts,
    }


def check_data_freshness(db: Session) -> dict:
    """Check that data pipelines ran recently."""
    alerts = []

    result = db.execute(text("""
        SELECT MAX(date) FROM market_bars_daily
    """))
    latest_bar = result.scalar()

    result = db.execute(text("""
        SELECT MAX(target_session_date) FROM features_snapshot
    """))
    latest_feature = result.scalar()

    result = db.execute(text("""
        SELECT MAX(as_of_time) FROM predictions
    """))
    latest_prediction = result.scalar()

    now = datetime.now(UTC)

    if latest_bar:
        bar_age_days = (date.today() - latest_bar).days
        if bar_age_days > 2:
            alerts.append({
                "level": "warning",
                "check": "stale_market_data",
                "detail": f"Latest market bar is {bar_age_days} days old ({latest_bar})",
            })
    else:
        bar_age_days = None

    if latest_prediction:
        pred_age_hours = (now - latest_prediction).total_seconds() / 3600
        if pred_age_hours > ALERT_THRESHOLDS["data_latency_hours"]:
            alerts.append({
                "level": "warning",
                "check": "stale_predictions",
                "detail": f"Latest prediction is {pred_age_hours:.1f}h old",
            })
    else:
        pred_age_hours = None

    return {
        "latest_market_bar": str(latest_bar) if latest_bar else None,
        "latest_feature": str(latest_feature) if latest_feature else None,
        "latest_prediction": str(latest_prediction) if latest_prediction else None,
        "alerts": alerts,
    }


def check_model_performance(db: Session) -> dict:
    """Check recent prediction accuracy against realized outcomes."""
    alerts = []

    result = db.execute(text("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN direction = realized_direction THEN 1 ELSE 0 END) AS correct
        FROM predictions
        WHERE realized_direction IS NOT NULL
          AND target_date >= CURRENT_DATE - 30
    """))
    row = result.fetchone()
    total = row[0] or 0
    correct = row[1] or 0

    accuracy = correct / total if total > 0 else None

    if accuracy is not None and accuracy > 0.65:
        alerts.append({
            "level": "critical",
            "check": "leakage_suspect",
            "detail": f"30d accuracy {accuracy:.4f} > 0.65 (leakage audit required)",
        })
    elif accuracy is not None and accuracy < 0.48:
        alerts.append({
            "level": "warning",
            "check": "model_degraded",
            "detail": f"30d accuracy {accuracy:.4f} < 0.48 (below random)",
        })

    return {
        "predictions_with_outcome_30d": total,
        "accuracy_30d": round(accuracy, 4) if accuracy is not None else None,
        "alerts": alerts,
    }


def check_signal_environment(db: Session) -> dict:
    """Monitor rolling live IC and cross-sectional dispersion."""
    alerts = []

    # Daily cross-sectional IC from realized prediction returns.
    result = db.execute(text("""
        SELECT target_date, probability_up, realized_return
        FROM predictions
        WHERE realized_return IS NOT NULL
          AND target_date >= CURRENT_DATE - 120
        ORDER BY target_date
    """))
    rows = result.fetchall()
    ic_rolling_60d = None
    n_ic_days = 0
    if rows:
        import pandas as pd

        pred_df = pd.DataFrame(rows, columns=["target_date", "probability_up", "realized_return"])
        daily_ic = []
        for dt, grp in pred_df.groupby("target_date"):
            if len(grp) < 10:
                continue
            ic = grp["probability_up"].corr(grp["realized_return"], method="spearman")
            if pd.notna(ic):
                daily_ic.append({"date": dt, "ic": float(ic)})
        if daily_ic:
            ic_df = pd.DataFrame(daily_ic).sort_values("date")
            n_ic_days = len(ic_df)
            if len(ic_df) >= 60:
                ic_rolling_60d = float(ic_df["ic"].tail(60).mean())
            else:
                ic_rolling_60d = float(ic_df["ic"].mean())

    if ic_rolling_60d is not None and ic_rolling_60d < 0.01:
        alerts.append({
            "level": "warning",
            "check": "low_live_rolling_ic",
            "detail": f"rolling live IC {ic_rolling_60d:.4f} < 0.0100",
        })

    # Cross-sectional dispersion of daily returns (last 60 sessions).
    result = db.execute(text("""
        WITH rets AS (
            SELECT
                date,
                symbol,
                close / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0) - 1 AS ret
            FROM market_bars_daily
            WHERE date >= CURRENT_DATE - 120
        )
        SELECT date, STDDEV(ret) AS dispersion
        FROM rets
        WHERE ret IS NOT NULL
        GROUP BY date
        ORDER BY date
    """))
    disp_rows = result.fetchall()
    dispersion_60d_mean = None
    if disp_rows:
        import pandas as pd

        disp_df = pd.DataFrame(disp_rows, columns=["date", "dispersion"])
        if not disp_df.empty:
            tail = disp_df.tail(60)
            dispersion_60d_mean = float(tail["dispersion"].mean())

    # Latest VIX proxy from features.
    result = db.execute(text("""
        SELECT AVG(vix_level)
        FROM features_snapshot
        WHERE target_session_date = (SELECT MAX(target_session_date) FROM features_snapshot)
    """))
    latest_vix = result.scalar()
    latest_vix = float(latest_vix) if latest_vix is not None else None

    return {
        "rolling_live_ic_60d": round(ic_rolling_60d, 5) if ic_rolling_60d is not None else None,
        "live_ic_days": n_ic_days,
        "dispersion_60d_mean": round(dispersion_60d_mean, 6) if dispersion_60d_mean is not None else None,
        "latest_vix_proxy": round(latest_vix, 3) if latest_vix is not None else None,
        "alerts": alerts,
    }


def check_feature_drift(db: Session) -> dict:
    """Check for feature distribution drift using simple PSI approximation.

    Compares recent 7-day feature distributions against the 30-day baseline.
    """
    alerts = []
    drift_scores = {}

    feature_cols = [
        "rsi_14", "momentum_5d", "rolling_volatility_20d", "vix_level"
    ]

    for col in feature_cols:
        result = db.execute(text(f"""
            SELECT
                AVG(CASE WHEN target_session_date >= CURRENT_DATE - 7 THEN {col} END) AS recent_mean,
                STDDEV(CASE WHEN target_session_date >= CURRENT_DATE - 7 THEN {col} END) AS recent_std,
                AVG(CASE WHEN target_session_date < CURRENT_DATE - 7
                    AND target_session_date >= CURRENT_DATE - 37 THEN {col} END) AS baseline_mean,
                STDDEV(CASE WHEN target_session_date < CURRENT_DATE - 7
                    AND target_session_date >= CURRENT_DATE - 37 THEN {col} END) AS baseline_std
            FROM features_snapshot
            WHERE target_session_date >= CURRENT_DATE - 37
        """))
        row = result.fetchone()

        if all(v is not None and v != 0 for v in row):
            recent_mean, recent_std, baseline_mean, baseline_std = [float(v) for v in row]
            psi = _approximate_psi(baseline_mean, baseline_std, recent_mean, recent_std)
            drift_scores[col] = round(psi, 4)

            if psi > ALERT_THRESHOLDS["psi_critical"]:
                alerts.append({
                    "level": "critical",
                    "check": f"drift_{col}",
                    "detail": f"{col} PSI={psi:.4f} > {ALERT_THRESHOLDS['psi_critical']}",
                })
            elif psi > ALERT_THRESHOLDS["psi_warning"]:
                alerts.append({
                    "level": "warning",
                    "check": f"drift_{col}",
                    "detail": f"{col} PSI={psi:.4f} > {ALERT_THRESHOLDS['psi_warning']}",
                })

    return {"drift_scores": drift_scores, "alerts": alerts}


def _approximate_psi(
    baseline_mean: float, baseline_std: float,
    recent_mean: float, recent_std: float,
    n_bins: int = 10,
) -> float:
    """Approximate PSI from summary statistics using normal assumption."""
    if baseline_std == 0 or recent_std == 0:
        return 0.0

    edges = np.linspace(
        min(baseline_mean - 3 * baseline_std, recent_mean - 3 * recent_std),
        max(baseline_mean + 3 * baseline_std, recent_mean + 3 * recent_std),
        n_bins + 1,
    )

    from scipy.stats import norm

    baseline_probs = np.diff(norm.cdf(edges, baseline_mean, baseline_std))
    recent_probs = np.diff(norm.cdf(edges, recent_mean, recent_std))

    baseline_probs = np.clip(baseline_probs, 1e-6, None)
    recent_probs = np.clip(recent_probs, 1e-6, None)

    baseline_probs /= baseline_probs.sum()
    recent_probs /= recent_probs.sum()

    psi = np.sum((recent_probs - baseline_probs) * np.log(recent_probs / baseline_probs))
    return float(psi)


def check_pipeline_health(db: Session) -> dict:
    """Check overall pipeline execution health."""
    alerts = []

    result = db.execute(text("""
        SELECT
            (SELECT COUNT(DISTINCT date) FROM market_bars_daily WHERE date >= CURRENT_DATE - 7) AS bar_days,
            (SELECT COUNT(DISTINCT target_session_date) FROM features_snapshot WHERE target_session_date >= CURRENT_DATE - 7) AS feat_days,
            (SELECT COUNT(DISTINCT target_date) FROM predictions WHERE target_date >= CURRENT_DATE - 7) AS pred_days
    """))
    row = result.fetchone()
    bar_days = row[0] or 0
    feat_days = row[1] or 0
    pred_days = row[2] or 0

    if bar_days > 0 and feat_days == 0:
        alerts.append({
            "level": "critical",
            "check": "feature_pipeline_stalled",
            "detail": f"Market data has {bar_days} days but features have {feat_days} days in last 7d",
        })

    if feat_days > 0 and pred_days == 0:
        alerts.append({
            "level": "critical",
            "check": "prediction_pipeline_stalled",
            "detail": f"Features have {feat_days} days but predictions have {pred_days} days in last 7d",
        })

    return {
        "market_bar_days_7d": bar_days,
        "feature_days_7d": feat_days,
        "prediction_days_7d": pred_days,
        "alerts": alerts,
    }
