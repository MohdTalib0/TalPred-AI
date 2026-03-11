"""Data contract validation utilities.

Shared validation logic used across all ingestion pipelines.
"""

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.models.schema import QuarantineRecord

logger = logging.getLogger(__name__)


def check_nulls(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Returns rows that have nulls in any required column."""
    mask = df[required_columns].isnull().any(axis=1)
    return df[mask]


def check_outliers(series: pd.Series, z_threshold: float = 5.0) -> pd.Series:
    """Returns boolean mask of outlier rows using robust z-score (MAD-based)."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (series - median) / mad
    return np.abs(modified_z) > z_threshold


def quarantine_records(
    db: Session,
    source: str,
    records: list[dict],
    reason: str,
) -> int:
    """Send invalid records to quarantine table."""
    count = 0
    for record in records:
        serializable = {}
        for k, v in record.items():
            if isinstance(v, (datetime, pd.Timestamp)):
                serializable[k] = v.isoformat()
            elif isinstance(v, (np.integer, np.floating)):
                serializable[k] = v.item()
            else:
                serializable[k] = v

        db.add(QuarantineRecord(
            source=source,
            record_data=serializable,
            failure_reason=reason,
        ))
        count += 1

    if count > 0:
        db.commit()
        logger.warning(f"Quarantined {count} records from {source}: {reason}")

    return count


def compute_freshness_check(
    latest_timestamp: datetime | None,
    sla_hours: float = 30.0,
) -> dict:
    """Check if data is fresh enough relative to SLA."""
    if latest_timestamp is None:
        return {"fresh": False, "lag_hours": None, "message": "no data found"}

    now = datetime.now(UTC).replace(tzinfo=None)
    lag = (now - latest_timestamp).total_seconds() / 3600

    return {
        "fresh": lag <= sla_hours,
        "lag_hours": round(lag, 2),
        "message": f"lag={round(lag, 2)}h, sla={sla_hours}h",
    }
