"""Macro data connector using FRED API."""

import logging
from datetime import UTC, date, datetime

import pandas as pd
from fredapi import Fred

from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SERIES = [
    ("DFF", "Federal Funds Rate"),
    ("CPIAUCSL", "CPI"),
    ("DCOILWTICO", "WTI Crude Oil"),
    ("DTWEXBGS", "Trade Weighted USD"),
    ("VIXCLS", "VIX"),
]


def get_fred_client() -> Fred | None:
    if not settings.fred_api_key:
        logger.warning("FRED_API_KEY not set")
        return None
    return Fred(api_key=settings.fred_api_key)


def fetch_macro_series(
    series_id: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Fetch a single FRED series. Returns list of observation dicts."""
    client = get_fred_client()
    if client is None:
        return []

    try:
        data = client.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
    except Exception:
        logger.exception(f"Failed to fetch FRED series {series_id}")
        return []

    if data is None or data.empty:
        return []

    now = datetime.now(UTC)
    records = []
    for obs_date, value in data.items():
        if pd.isna(value):
            continue
        records.append({
            "series_id": series_id,
            "observation_date": obs_date.date(),
            "value": float(value),
            "release_time_utc": now,
            "available_at_utc": now,
            "source": "fred",
        })

    return records


def fetch_all_default_series(
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Fetch all default macro series."""
    all_records = []
    for series_id, name in DEFAULT_SERIES:
        logger.info(f"Fetching {name} ({series_id})")
        records = fetch_macro_series(series_id, start_date, end_date)
        all_records.extend(records)
        logger.info(f"  -> {len(records)} observations")
    return all_records
