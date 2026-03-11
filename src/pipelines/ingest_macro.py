"""Macro series ingestion pipeline.

Fetches FRED macro indicators and upserts into macro_series table.
"""

import logging
from datetime import date

from sqlalchemy.orm import Session

from src.connectors.macro import fetch_all_default_series
from src.models.schema import MacroSeries, QuarantineRecord

logger = logging.getLogger(__name__)


def validate_macro_record(record: dict) -> tuple[bool, str]:
    if record.get("value") is None:
        return False, "missing value"
    if not record.get("series_id"):
        return False, "missing series_id"
    if not record.get("observation_date"):
        return False, "missing observation_date"
    return True, ""


def ingest_macro(
    db: Session,
    start_date: date,
    end_date: date,
) -> dict:
    """Ingest all default FRED macro series."""
    records = fetch_all_default_series(start_date, end_date)

    inserted = 0
    updated = 0
    quarantined = 0

    for record in records:
        is_valid, reason = validate_macro_record(record)
        if not is_valid:
            db.add(QuarantineRecord(
                source="macro_series",
                record_data=record,
                failure_reason=reason,
            ))
            quarantined += 1
            continue

        existing = (
            db.query(MacroSeries)
            .filter(
                MacroSeries.series_id == record["series_id"],
                MacroSeries.observation_date == record["observation_date"],
            )
            .first()
        )

        if existing:
            existing.value = record["value"]
            existing.release_time_utc = record.get("release_time_utc")
            existing.available_at_utc = record.get("available_at_utc")
            updated += 1
        else:
            db.add(MacroSeries(**record))
            inserted += 1

    db.commit()
    logger.info(f"Macro: inserted={inserted}, updated={updated}, quarantined={quarantined}")
    return {"inserted": inserted, "updated": updated, "quarantined": quarantined}
