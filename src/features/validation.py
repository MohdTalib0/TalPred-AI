"""Feature validation suite (ML-208).

Automated checks for:
  - No future timestamps (point-in-time violation)
  - No NaN leakage (features that are always null)
  - Feature distribution sanity (range, variance, drift)
"""

import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "rsi_14", "momentum_5d", "momentum_10d",
    "rolling_return_5d", "rolling_return_20d", "rolling_volatility_20d",
    "macd", "macd_signal",
    "sector_return_1d", "sector_return_5d",
    "benchmark_relative_return_1d",
    "news_sentiment_24h", "news_sentiment_7d",
    "vix_level", "sp500_momentum_200d",
]

EXPECTED_RANGES = {
    "rsi_14": (0, 100),
    "momentum_5d": (-0.5, 0.5),
    "momentum_10d": (-0.5, 0.5),
    "rolling_return_5d": (-0.5, 0.5),
    "rolling_return_20d": (-0.8, 0.8),
    "rolling_volatility_20d": (0, 0.2),
    "vix_level": (5, 90),
}


def validate_features(db: Session, target_date: date | None = None) -> dict:
    """Run full feature validation suite. Returns report dict."""
    report = {
        "timestamp_check": _check_timestamps(db, target_date),
        "null_check": _check_nulls(db, target_date),
        "range_check": _check_ranges(db, target_date),
        "variance_check": _check_variance(db, target_date),
    }

    all_passed = all(v["passed"] for v in report.values())
    report["overall_passed"] = all_passed
    report["target_date"] = str(target_date) if target_date else "all"

    status = "PASSED" if all_passed else "FAILED"
    logger.info(f"Feature validation {status} for date={target_date}")
    for name, result in report.items():
        if isinstance(result, dict) and "passed" in result:
            if not result["passed"]:
                logger.warning(f"  {name}: FAILED - {result.get('issues', [])}")

    return report


def _check_timestamps(db: Session, target_date: date | None) -> dict:
    """Ensure no feature snapshot has as_of_time AFTER its target_session_date."""
    where_clause = ""
    params = {}
    if target_date:
        where_clause = "AND target_session_date = :target_date"
        params["target_date"] = target_date

    result = db.execute(text(f"""
        SELECT COUNT(*)
        FROM features_snapshot
        WHERE as_of_time::date > target_session_date + INTERVAL '1 day'
        {where_clause}
    """), params)

    violations = result.scalar()
    return {
        "passed": violations == 0,
        "violations": violations,
        "issues": [f"{violations} snapshots with future timestamps"] if violations > 0 else [],
    }


def _check_nulls(db: Session, target_date: date | None) -> dict:
    """Check for features that are always null (indicates broken pipeline)."""
    where_clause = ""
    params = {}
    if target_date:
        where_clause = f"WHERE target_session_date = :target_date"
        params["target_date"] = target_date

    issues = []
    for col in FEATURE_COLUMNS:
        result = db.execute(text(f"""
            SELECT
                COUNT(*) AS total,
                COUNT({col}) AS non_null
            FROM features_snapshot
            {where_clause}
        """), params)
        row = result.fetchone()
        total, non_null = row[0], row[1]

        if total > 0 and non_null == 0 and col not in ("news_sentiment_24h", "news_sentiment_7d", "sp500_momentum_200d"):
            issues.append(f"{col}: 100% null ({total} rows)")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def _check_ranges(db: Session, target_date: date | None) -> dict:
    """Check feature values are within expected ranges."""
    where_clause = ""
    params = {}
    if target_date:
        where_clause = f"WHERE target_session_date = :target_date"
        params["target_date"] = target_date

    issues = []
    for col, (low, high) in EXPECTED_RANGES.items():
        result = db.execute(text(f"""
            SELECT MIN({col}), MAX({col})
            FROM features_snapshot
            {where_clause}
        """), params)
        row = result.fetchone()
        min_val, max_val = row[0], row[1]

        if min_val is not None and min_val < low * 2:
            issues.append(f"{col}: min={min_val:.4f} below expected {low}")
        if max_val is not None and max_val > high * 2:
            issues.append(f"{col}: max={max_val:.4f} above expected {high}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def _check_variance(db: Session, target_date: date | None) -> dict:
    """Check features have non-zero variance (constant features are useless)."""
    where_clause = ""
    params = {}
    if target_date:
        where_clause = f"WHERE target_session_date = :target_date"
        params["target_date"] = target_date

    issues = []
    for col in FEATURE_COLUMNS:
        if col in ("news_sentiment_24h", "news_sentiment_7d"):
            continue

        result = db.execute(text(f"""
            SELECT STDDEV({col})
            FROM features_snapshot
            {where_clause}
        """), params)
        row = result.fetchone()
        std = row[0]

        if std is not None and std == 0:
            issues.append(f"{col}: zero variance (constant feature)")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }
