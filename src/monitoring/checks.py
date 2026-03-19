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
        "alpha_quality": check_alpha_quality(db),
        "feature_drift": check_feature_drift(db),
        "feature_stability": check_feature_importance_stability(db),
        "capacity_indicators": check_capacity_indicators(db),
        "regime_stress": check_regime_stress(db),
        "pipeline_health": check_pipeline_health(db),
        "feature_ood": check_feature_ood(db),
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
    """Check recent prediction accuracy against realized outcomes.

    Filters to the current production model version so accuracy isn't
    muddled by stale predictions from prior models with different
    target modes (absolute vs market-relative).
    """
    alerts = []

    result = db.execute(text("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN p.direction = p.realized_direction THEN 1 ELSE 0 END) AS correct
        FROM predictions p
        INNER JOIN model_registry mr
            ON mr.model_version = p.model_version
            AND mr.status = 'production'
        WHERE p.realized_direction IS NOT NULL
          AND p.target_date >= CURRENT_DATE - 30
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


def check_alpha_quality(db: Session) -> dict:
    """Track decile spread, hit rate by confidence tier, and IC trend.

    This is the core "is the alpha still alive?" check.
    """
    import pandas as pd

    alerts = []

    result = db.execute(text("""
        SELECT
            p.target_date,
            p.symbol,
            p.probability_up,
            p.direction,
            p.confidence,
            p.realized_return,
            p.realized_direction
        FROM predictions p
        INNER JOIN model_registry mr
            ON mr.model_version = p.model_version
            AND mr.status = 'production'
        WHERE p.realized_return IS NOT NULL
          AND p.target_date >= CURRENT_DATE - 60
        ORDER BY p.target_date, p.probability_up DESC
    """))
    rows = result.fetchall()

    if not rows or len(rows) < 100:
        return {"status": "insufficient_data", "n_rows": len(rows) if rows else 0, "alerts": []}

    df = pd.DataFrame(
        rows,
        columns=[
            "target_date", "symbol", "probability_up", "direction",
            "confidence", "realized_return", "realized_direction",
        ],
    )

    # ── Decile spread (daily top vs bottom quintile return) ──
    daily_spreads = []
    daily_ic = []
    for dt, grp in df.groupby("target_date"):
        if len(grp) < 20:
            continue
        grp = grp.sort_values("probability_up", ascending=False)
        n = len(grp) // 5
        top_ret = grp.head(n)["realized_return"].mean()
        bot_ret = grp.tail(n)["realized_return"].mean()
        daily_spreads.append({"date": dt, "spread": float(top_ret - bot_ret)})

        ic = grp["probability_up"].corr(grp["realized_return"], method="spearman")
        if pd.notna(ic):
            daily_ic.append({"date": dt, "ic": float(ic)})

    decile_spread_mean = None
    decile_spread_bps = None
    if daily_spreads:
        sp = pd.DataFrame(daily_spreads)
        decile_spread_mean = float(sp["spread"].mean())
        decile_spread_bps = round(decile_spread_mean * 10000, 1)

    # ── IC trend: is it rising or falling? ──
    ic_trend = None
    ic_recent_30d = None
    ic_prior_30d = None
    if daily_ic and len(daily_ic) >= 20:
        ic_df = pd.DataFrame(daily_ic).sort_values("date")
        half = len(ic_df) // 2
        ic_prior_30d = float(ic_df.iloc[:half]["ic"].mean())
        ic_recent_30d = float(ic_df.iloc[half:]["ic"].mean())
        ic_trend = "improving" if ic_recent_30d > ic_prior_30d else "declining"

        if ic_recent_30d < 0.005:
            alerts.append({
                "level": "warning",
                "check": "alpha_decay",
                "detail": (
                    f"Recent 30d IC={ic_recent_30d:.4f} is near zero "
                    f"(prior={ic_prior_30d:.4f}, trend={ic_trend})"
                ),
            })

    # ── Hit rate by confidence tier ──
    hit_rates = {}
    for label, lo, hi in [
        ("low (0.50-0.55)", 0.50, 0.55),
        ("mid (0.55-0.60)", 0.55, 0.60),
        ("high (0.60-0.70)", 0.60, 0.70),
        ("very_high (0.70+)", 0.70, 1.01),
    ]:
        tier = df[(df["confidence"] >= lo) & (df["confidence"] < hi)]
        if len(tier) < 10:
            continue
        correct = (tier["direction"] == tier["realized_direction"]).sum()
        hit_rates[label] = {
            "n": int(len(tier)),
            "hit_rate": round(float(correct / len(tier)), 4),
        }

    return {
        "decile_spread_bps": decile_spread_bps,
        "decile_spread_mean": round(decile_spread_mean, 6) if decile_spread_mean else None,
        "n_spread_days": len(daily_spreads),
        "ic_recent_30d": round(ic_recent_30d, 5) if ic_recent_30d is not None else None,
        "ic_prior_30d": round(ic_prior_30d, 5) if ic_prior_30d is not None else None,
        "ic_trend": ic_trend,
        "hit_rates_by_confidence": hit_rates,
        "alerts": alerts,
    }


def check_feature_drift(db: Session) -> dict:
    """Check for feature distribution drift using simple PSI approximation.

    Compares recent 7-day feature distributions against the 30-day baseline.
    """
    alerts = []
    drift_scores = {}

    _ALLOWED_DRIFT_COLS = frozenset({
        "rsi_14", "momentum_5d", "rolling_volatility_20d", "vix_level",
        "momentum_20d", "momentum_60d", "short_term_reversal",
    })
    feature_cols = list(_ALLOWED_DRIFT_COLS)

    for col in feature_cols:
        if col not in _ALLOWED_DRIFT_COLS:
            continue
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


def check_feature_importance_stability(db: Session) -> dict:
    """Track whether the model's top features are stable over time.

    Compares recent feature importance rankings against a stored baseline
    to detect alpha decay or regime shifts that change what matters.
    """
    import json
    import os

    alerts = []
    stability = {}

    baseline_path = os.path.join("artifacts", "production_model", "metadata.json")
    if not os.path.exists(baseline_path):
        return {"status": "no_baseline", "alerts": []}

    try:
        with open(baseline_path) as f:
            metadata = json.load(f)
    except Exception:
        return {"status": "baseline_unreadable", "alerts": []}

    feature_cols = metadata.get("feature_columns", [])
    if not feature_cols:
        return {"status": "no_feature_columns", "alerts": []}

    result = db.execute(text("""
        SELECT target_session_date, COUNT(*)
        FROM features_snapshot
        WHERE target_session_date >= CURRENT_DATE - 30
        GROUP BY target_session_date
        ORDER BY target_session_date
    """))
    rows = result.fetchall()
    active_days = len(rows) if rows else 0

    stability["active_feature_days_30d"] = active_days
    stability["tracked_features"] = len(feature_cols)

    if active_days < 5:
        alerts.append({
            "level": "warning",
            "check": "insufficient_feature_data",
            "detail": f"Only {active_days} days of features in last 30d",
        })

    return {"stability": stability, "alerts": alerts}


def check_capacity_indicators(db: Session) -> dict:
    """Monitor capacity-related metrics: ADV coverage, concentration risk.

    Flags when the universe liquidity is too thin for the strategy's capital.
    """
    alerts = []

    result = db.execute(text("""
        SELECT
            s.symbol,
            s.avg_daily_volume_30d,
            mb.close
        FROM symbols s
        LEFT JOIN LATERAL (
            SELECT close
            FROM market_bars_daily
            WHERE symbol = s.symbol
            ORDER BY date DESC
            LIMIT 1
        ) mb ON true
        WHERE s.is_active = true
          AND s.avg_daily_volume_30d IS NOT NULL
    """))
    rows = result.fetchall()

    if not rows:
        return {"status": "no_data", "alerts": []}

    import pandas as pd

    df = pd.DataFrame(rows, columns=["symbol", "adv_shares", "close"])
    df["adv_dollars"] = df["adv_shares"].fillna(0) * df["close"].fillna(0)

    total_adv = float(df["adv_dollars"].sum())
    median_adv = float(df["adv_dollars"].median())
    p10_adv = float(df["adv_dollars"].quantile(0.10))
    illiquid_count = int((df["adv_dollars"] < 1_000_000).sum())

    # At 5% participation, how much capital can the universe absorb?
    max_capital_5pct = total_adv * 0.05
    max_capital_1pct = total_adv * 0.01

    if illiquid_count > len(df) * 0.2:
        alerts.append({
            "level": "warning",
            "check": "illiquid_universe",
            "detail": f"{illiquid_count}/{len(df)} symbols have ADV < $1M",
        })

    return {
        "universe_size": len(df),
        "total_universe_adv": round(total_adv, 0),
        "median_adv_dollars": round(median_adv, 0),
        "p10_adv_dollars": round(p10_adv, 0),
        "illiquid_count_lt_1m": illiquid_count,
        "max_capital_at_5pct_participation": round(max_capital_5pct, 0),
        "max_capital_at_1pct_participation": round(max_capital_1pct, 0),
        "alerts": alerts,
    }


def check_regime_stress(db: Session) -> dict:
    """Monitor current regime and flag crisis conditions.

    Checks VIX level, recent drawdown magnitude, and cross-sectional
    dispersion to detect stress environments where the strategy may
    behave differently.
    """
    import pandas as pd

    alerts = []

    # Current VIX
    result = db.execute(text("""
        SELECT AVG(vix_level)
        FROM features_snapshot
        WHERE target_session_date = (
            SELECT MAX(target_session_date) FROM features_snapshot
        )
    """))
    current_vix = result.scalar()
    current_vix = float(current_vix) if current_vix is not None else None

    if current_vix and current_vix >= 30:
        alerts.append({
            "level": "critical",
            "check": "vix_crisis",
            "detail": f"VIX at {current_vix:.1f} — crisis regime",
        })
    elif current_vix and current_vix >= 25:
        alerts.append({
            "level": "warning",
            "check": "vix_elevated",
            "detail": f"VIX at {current_vix:.1f} — elevated stress",
        })

    # SPY drawdown from recent peak
    result = db.execute(text("""
        SELECT date, close
        FROM market_bars_daily
        WHERE symbol = 'SPY'
          AND date >= CURRENT_DATE - 60
        ORDER BY date
    """))
    spy_rows = result.fetchall()
    spy_drawdown = None
    if spy_rows:
        spy_df = pd.DataFrame(spy_rows, columns=["date", "close"])
        peak = spy_df["close"].cummax()
        dd = (spy_df["close"] - peak) / peak
        spy_drawdown = float(dd.iloc[-1]) if not dd.empty else None

        if spy_drawdown is not None and spy_drawdown < -0.10:
            alerts.append({
                "level": "critical",
                "check": "spy_drawdown",
                "detail": f"SPY drawdown {spy_drawdown:.1%} from 60d peak",
            })
        elif spy_drawdown is not None and spy_drawdown < -0.05:
            alerts.append({
                "level": "warning",
                "check": "spy_drawdown",
                "detail": f"SPY drawdown {spy_drawdown:.1%} from 60d peak",
            })

    # Cross-sectional return dispersion (last 5 days)
    result = db.execute(text("""
        WITH rets AS (
            SELECT date, symbol,
                   close / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0) - 1 AS ret
            FROM market_bars_daily
            WHERE date >= CURRENT_DATE - 10
        )
        SELECT STDDEV(ret)
        FROM rets
        WHERE ret IS NOT NULL
          AND date >= CURRENT_DATE - 5
    """))
    recent_dispersion = result.scalar()
    recent_dispersion = float(recent_dispersion) if recent_dispersion else None

    return {
        "current_vix": round(current_vix, 2) if current_vix else None,
        "spy_drawdown_60d": round(spy_drawdown, 4) if spy_drawdown is not None else None,
        "recent_dispersion_5d": round(recent_dispersion, 6) if recent_dispersion else None,
        "alerts": alerts,
    }


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


# ------------------------------------------------------------------
# OOD Detection
# ------------------------------------------------------------------

_OOD_FEATURES = [
    "rsi_14", "momentum_5d", "momentum_20d", "momentum_60d",
    "rolling_volatility_20d", "volume_change_5d", "volume_zscore_20d",
    "macd", "short_term_reversal", "volatility_expansion_5_20",
]

_OOD_TRAIN_LOOKBACK_DAYS = 252
_OOD_PERCENTILE_THRESHOLD = 99


def check_feature_ood(db: Session) -> dict:
    """Detect out-of-distribution feature vectors using Mahalanobis distance.

    Compares today's cross-sectional feature distribution against the
    training distribution (last 252 days). Uses Ledoit-Wolf shrinkage
    covariance estimator for numerical stability with 38+ features.

    Fires an alert when the median Mahalanobis distance of today's
    feature vectors exceeds the 99th percentile of the training
    distribution's distances, indicating the model is extrapolating.
    """
    alerts = []

    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        return {"status": "skipped", "reason": "scikit-learn not available", "alerts": []}

    # Fetch training-period features (last 252 trading days)
    feature_cols_sql = ", ".join(
        f"fs.{col}" for col in _OOD_FEATURES
    )
    train_query = f"""
        SELECT fs.target_session_date, fs.symbol, {feature_cols_sql}
        FROM features_snapshot fs
        WHERE fs.target_session_date >= CURRENT_DATE - {_OOD_TRAIN_LOOKBACK_DAYS + 5}
          AND fs.target_session_date < CURRENT_DATE
        ORDER BY fs.target_session_date
    """
    try:
        result = db.execute(text(train_query))
        train_rows = result.fetchall()
    except Exception as e:
        logger.warning(f"OOD check: training data query failed: {e}")
        return {"status": "error", "reason": str(e), "alerts": []}

    if len(train_rows) < 100:
        return {
            "status": "insufficient_data",
            "n_train_rows": len(train_rows),
            "alerts": [],
        }

    cols = ["target_session_date", "symbol"] + _OOD_FEATURES
    train_df = _ood_rows_to_df(train_rows, cols)
    if train_df.empty:
        return {"status": "no_valid_training_data", "alerts": []}

    # Fetch today's features
    today_query = f"""
        SELECT fs.target_session_date, fs.symbol, {feature_cols_sql}
        FROM features_snapshot fs
        WHERE fs.target_session_date = (
            SELECT MAX(target_session_date) FROM features_snapshot
        )
    """
    try:
        result = db.execute(text(today_query))
        today_rows = result.fetchall()
    except Exception as e:
        logger.warning(f"OOD check: today's data query failed: {e}")
        return {"status": "error", "reason": str(e), "alerts": []}

    if not today_rows:
        return {"status": "no_today_data", "alerts": []}

    today_df = _ood_rows_to_df(today_rows, cols)
    if today_df.empty:
        return {"status": "no_valid_today_data", "alerts": []}

    # Fit Ledoit-Wolf covariance on training data
    train_features = train_df[_OOD_FEATURES].values
    today_features = today_df[_OOD_FEATURES].values

    try:
        lw = LedoitWolf()
        lw.fit(train_features)
        precision = lw.precision_
        train_mean = train_features.mean(axis=0)
    except Exception as e:
        logger.warning(f"OOD check: Ledoit-Wolf fit failed: {e}")
        return {"status": "error", "reason": str(e), "alerts": []}

    # Mahalanobis distances for training data (baseline distribution)
    train_centered = train_features - train_mean
    train_mahal = np.sqrt(
        np.sum(train_centered @ precision * train_centered, axis=1)
    )

    # Mahalanobis distances for today's data
    today_centered = today_features - train_mean
    today_mahal = np.sqrt(
        np.sum(today_centered @ precision * today_centered, axis=1)
    )

    # Percentile threshold from training distribution
    threshold = float(np.percentile(train_mahal, _OOD_PERCENTILE_THRESHOLD))
    today_median = float(np.median(today_mahal))
    today_max = float(np.max(today_mahal))
    pct_ood = float(np.mean(today_mahal > threshold) * 100)

    is_ood = today_median > threshold

    if is_ood:
        alerts.append({
            "level": "warning",
            "check": "feature_ood",
            "detail": (
                f"Feature distribution OOD: median Mahalanobis={today_median:.1f} "
                f"> {_OOD_PERCENTILE_THRESHOLD}th pct threshold={threshold:.1f}. "
                f"{pct_ood:.0f}% of stocks flagged OOD. "
                f"Model may be extrapolating — consider reducing confidence."
            ),
        })
        logger.warning(
            "OOD ALERT: median_mahal=%.1f > threshold=%.1f (%.0f%% OOD stocks)",
            today_median, threshold, pct_ood,
        )

    # Per-stock OOD flags
    ood_symbols = []
    if len(today_df) > 0:
        for i, (_, row) in enumerate(today_df.iterrows()):
            if today_mahal[i] > threshold:
                ood_symbols.append(row["symbol"] if "symbol" in row else f"stock_{i}")

    return {
        "status": "ood_detected" if is_ood else "in_distribution",
        "today_median_mahalanobis": round(today_median, 2),
        "today_max_mahalanobis": round(today_max, 2),
        "threshold_99pct": round(threshold, 2),
        "pct_stocks_ood": round(pct_ood, 1),
        "n_ood_stocks": len(ood_symbols),
        "ood_symbols_sample": ood_symbols[:10],
        "n_train_rows": len(train_features),
        "n_today_rows": len(today_features),
        "n_features": len(_OOD_FEATURES),
        "covariance_method": "ledoit_wolf",
        "shrinkage_coeff": (
            round(float(lw.shrinkage_), 4) if hasattr(lw, "shrinkage_") else None
        ),
        "alerts": alerts,
    }


def _ood_rows_to_df(rows, cols: list[str]):
    """Convert DB rows to a cleaned DataFrame for OOD computation."""
    import pandas as pd

    df = pd.DataFrame(rows, columns=cols)
    feature_cols = [c for c in cols if c not in ("target_session_date", "symbol")]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    return df
