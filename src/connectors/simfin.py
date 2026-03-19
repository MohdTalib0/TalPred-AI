"""SimFin API connector for quarterly financial statements.

Downloads and caches income statement (pl), balance sheet (bs), and
cash flow (cf) data from the SimFin API v3.

Follows the same connector convention as market.py, macro.py, news.py:
  - Handles API authentication, batching, HTTP errors
  - Returns validated DataFrames
  - Caches raw responses to disk (parquet)

Requires SIMFIN_API_KEY environment variable (free tier: 2000 requests/day).

Usage:
    from src.connectors.simfin import fetch_statements

    pl_df = fetch_statements(symbols, "pl", start_date, end_date)
"""

import hashlib
import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("artifacts") / "cache" / "simfin"
SIMFIN_BASE = "https://backend.simfin.com/api/v3"
CACHE_SCHEMA_VERSION = 1


def _get_api_key() -> str:
    key = os.environ.get("SIMFIN_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "SIMFIN_API_KEY not set. Get a free key at https://app.simfin.com/login"
        )
    return key


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.parquet"


def _cache_meta_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.meta.json"


def _is_cache_valid(cache_file: Path, meta_file: Path) -> bool:
    """Check cache exists and schema version matches."""
    if not cache_file.exists():
        return False
    if not meta_file.exists():
        return False
    import json
    try:
        with open(meta_file) as f:
            meta = json.load(f)
        return meta.get("schema_version") == CACHE_SCHEMA_VERSION
    except Exception:
        return False


def _write_cache_meta(meta_file: Path, extra: dict | None = None) -> None:
    import json
    from datetime import datetime, UTC
    meta = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cached_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        meta.update(extra)
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)


def fetch_statements(
    symbols: list[str],
    statement: str,
    start_date: date,
    end_date: date,
    batch_size: int = 15,
) -> pd.DataFrame:
    """Download financial statements from SimFin API (v3).

    Args:
        symbols: List of ticker symbols.
        statement: Statement type — "pl" (income), "bs" (balance sheet),
            or "cf" (cash flow).
        start_date: Start of fiscal year range.
        end_date: End of fiscal year range.
        batch_size: Tickers per API request (SimFin free tier limit).

    Returns:
        DataFrame with raw statement data. Columns vary by statement type.
        Includes 'symbol' column for identification.

    Falls back to cached data if the API is unavailable.
    """
    import requests

    api_key = _get_api_key()
    cache_key = hashlib.md5(
        f"{sorted(symbols)}_{statement}_{start_date}_{end_date}".encode()
    ).hexdigest()[:12]
    cache_name = f"simfin_{statement}_{cache_key}"
    cache_file = _cache_path(cache_name)
    meta_file = _cache_meta_path(cache_name)

    if _is_cache_valid(cache_file, meta_file):
        logger.info(f"Loading cached SimFin {statement} from {cache_file}")
        return pd.read_parquet(cache_file)

    all_rows = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        ticker_str = ",".join(batch)
        url = f"{SIMFIN_BASE}/companies/statements/compact"
        params = {
            "ticker": ticker_str,
            "statement": statement,
            "period": "quarters",
            "fyear": ",".join(
                str(y)
                for y in range(start_date.year - 1, end_date.year + 1)
            ),
            "api-key": api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"SimFin API error for batch {i}: {e}")
            continue

        for entry in data:
            ticker = entry.get("ticker", "")
            columns = entry.get("columns", [])
            for row in entry.get("data", []):
                row_dict = dict(zip(columns, row))
                row_dict["symbol"] = ticker
                all_rows.append(row_dict)

    if not all_rows:
        logger.warning(f"No SimFin {statement} data retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.to_parquet(cache_file, index=False)
    _write_cache_meta(meta_file, {
        "statement": statement,
        "n_rows": len(df),
        "n_symbols": len(df["symbol"].unique()),
    })
    logger.info(f"Cached {len(df)} SimFin {statement} rows → {cache_file}")
    return df
