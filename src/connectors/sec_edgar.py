"""SEC EDGAR XBRL JSON API connector — completely free, no API key.

Fetches structured quarterly financials with **actual SEC filing dates**,
providing true point-in-time (PIT) timestamps for as-of joins.

API endpoint:
  https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json

Rate limit: 10 requests/second (SEC guideline), we use 1 req/sec to be safe.

Coverage:
  - All US-listed companies with SEC filings
  - Historical data back to ~2009 for most companies
  - Fields: NetIncomeLoss, Revenues, Assets, StockholdersEquity,
    OperatingCashFlow, EarningsPerShareBasic, etc.

Usage:
    from src.connectors.sec_edgar import fetch_company_facts, extract_quarterly

    facts = fetch_company_facts("AAPL")
    df = extract_quarterly(facts, "AAPL")
"""

import json
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"
_CIK_LOOKUP_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&enddt=2025-01-01&forms=10-K,10-Q"
_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

CACHE_DIR = Path("artifacts") / "cache" / "edgar"
CIK_CACHE_FILE = CACHE_DIR / "_cik_map.json"

_RATE_LIMIT_SECONDS = 1.1
_last_request_time = 0.0

_HEADERS = {
    "User-Agent": "TalPredAI/1.0 (research@talpred.ai)",
    "Accept-Encoding": "gzip, deflate",
}

XBRL_FIELDS = {
    "us-gaap": {
        "NetIncomeLoss": "net_income",
        "Revenues": "total_revenue",
        "RevenueFromContractWithCustomerExcludingAssessedTax": "total_revenue_alt",
        "Assets": "total_assets",
        "StockholdersEquity": "stockholders_equity",
        "Liabilities": "total_liabilities",
        "OperatingIncomeLoss": "operating_income",
        "GrossProfit": "gross_profit",
        "CostOfRevenue": "cost_of_revenue",
        "EarningsPerShareBasic": "eps_basic",
        "EarningsPerShareDiluted": "eps_diluted",
        "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
        "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    },
}


def _rate_limit() -> None:
    """Enforce SEC rate limit."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def load_cik_map() -> dict[str, str]:
    """Load ticker → CIK mapping from SEC.

    Downloads once and caches locally. CIK numbers are zero-padded to 10 digits.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CIK_CACHE_FILE.exists():
        cache_age = (date.today() - date.fromtimestamp(CIK_CACHE_FILE.stat().st_mtime)).days
        if cache_age < 30:
            with open(CIK_CACHE_FILE) as f:
                return json.load(f)

    _rate_limit()
    try:
        resp = requests.get(_TICKER_CIK_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch CIK map from SEC: {e}")
        if CIK_CACHE_FILE.exists():
            with open(CIK_CACHE_FILE) as f:
                return json.load(f)
        return {}

    cik_map = {}
    for entry in data.values():
        ticker = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if ticker and cik:
            cik_map[ticker] = cik

    with open(CIK_CACHE_FILE, "w") as f:
        json.dump(cik_map, f)

    logger.info(f"CIK map loaded: {len(cik_map)} tickers")
    return cik_map


def fetch_company_facts(
    symbol: str,
    cik_map: dict[str, str] | None = None,
    use_cache: bool = True,
) -> dict | None:
    """Fetch XBRL company facts for a single symbol.

    Returns raw JSON dict from SEC EDGAR, or None on failure.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cik_map is None:
        cik_map = load_cik_map()

    cik = cik_map.get(symbol.upper())
    if not cik:
        logger.debug(f"{symbol}: CIK not found in SEC map")
        return None

    cache_file = CACHE_DIR / f"{symbol}_{cik}.json"
    if use_cache and cache_file.exists():
        cache_age = (date.today() - date.fromtimestamp(cache_file.stat().st_mtime)).days
        if cache_age < 7:
            with open(cache_file) as f:
                return json.load(f)

    url = f"{_BASE_URL}/CIK{cik}.json"
    _rate_limit()

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 404:
            logger.debug(f"{symbol} (CIK {cik}): no EDGAR data")
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"{symbol}: EDGAR fetch failed: {e}")
        return None

    with open(cache_file, "w") as f:
        json.dump(data, f)

    return data


def extract_quarterly(
    facts: dict,
    symbol: str,
) -> pd.DataFrame:
    """Extract quarterly financial data from EDGAR company facts.

    Filters to 10-Q and 10-K filings, extracts the actual SEC filing
    date ('filed') for true point-in-time joins.

    Returns DataFrame with columns:
        symbol, period_end_date, filed_at, + financial fields
    """
    if not facts:
        return pd.DataFrame()

    facts_data = facts.get("facts", {})
    records: dict[str, dict] = {}

    for taxonomy, fields in XBRL_FIELDS.items():
        tax_data = facts_data.get(taxonomy, {})

        for xbrl_field, our_field in fields.items():
            field_data = tax_data.get(xbrl_field, {})
            units = field_data.get("units", {})

            unit_key = "USD" if our_field != "eps_basic" and our_field != "eps_diluted" else "USD/shares"
            entries = units.get(unit_key, [])
            if not entries and units:
                entries = list(units.values())[0]

            for entry in entries:
                form = entry.get("form", "")
                if form not in ("10-Q", "10-K"):
                    continue

                end_date = entry.get("end")
                filed_date = entry.get("filed")
                val = entry.get("val")

                if not end_date or val is None:
                    continue

                key = f"{end_date}_{form}"
                if key not in records:
                    records[key] = {
                        "symbol": symbol,
                        "period_end_date": pd.to_datetime(end_date).date(),
                        "filed_at": (
                            pd.to_datetime(filed_date).date()
                            if filed_date else None
                        ),
                        "form_type": form,
                    }

                if our_field.endswith("_alt"):
                    base_field = our_field[:-4]
                    if base_field not in records[key] or records[key][base_field] is None:
                        records[key][base_field] = float(val)
                else:
                    records[key][our_field] = float(val)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(list(records.values()))
    df = df.sort_values("period_end_date").reset_index(drop=True)

    # Deduplicate: if both 10-Q and 10-K cover same period, prefer 10-K
    df = df.drop_duplicates(subset=["symbol", "period_end_date"], keep="last")

    return df


def fetch_fundamentals_edgar(
    symbols: list[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch quarterly fundamentals for multiple symbols via EDGAR.

    This is the main entry point. Returns a DataFrame with true PIT
    filing dates for all available symbols.
    """
    cik_map = load_cik_map()
    all_frames = []

    for symbol in symbols:
        facts = fetch_company_facts(symbol, cik_map=cik_map, use_cache=use_cache)
        if facts is None:
            continue

        df = extract_quarterly(facts, symbol)
        if not df.empty:
            all_frames.append(df)
            logger.debug(f"{symbol}: {len(df)} quarterly records from EDGAR")

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)
