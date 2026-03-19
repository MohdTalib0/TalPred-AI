"""Alpha Vantage free-tier market data connector.

Provides daily OHLCV as a fallback/validation source alongside yfinance.
Free tier: 25 requests/day, 5 requests/minute.

Used for:
  - Cross-source reconciliation (catches yfinance-specific split errors)
  - Gap-filling when yfinance returns incomplete data
  - NOT a primary source (rate limits too low for full universe)

Requires ALPHA_VANTAGE_API_KEY environment variable.
Free key: https://www.alphavantage.co/support/#api-key

Usage:
    from src.connectors.alpha_vantage import fetch_daily_bars

    df = fetch_daily_bars("AAPL", start_date, end_date)
"""

import logging
import os
import time
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"
_RATE_LIMIT_SECONDS = 12.5  # 5 requests/minute → 1 per 12s with buffer


def _get_api_key() -> str:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ALPHA_VANTAGE_API_KEY not set. "
            "Get a free key at https://www.alphavantage.co/support/#api-key"
        )
    return key


def fetch_daily_bars(
    symbol: str,
    start_date: date,
    end_date: date,
    outputsize: str = "full",
) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV from Alpha Vantage.

    Args:
        symbol: Ticker symbol.
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        outputsize: "compact" (last 100 days) or "full" (20+ years).

    Returns DataFrame matching market.py interface.
    """
    import requests

    api_key = _get_api_key()
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }

    try:
        resp = requests.get(_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")
        return pd.DataFrame()

    if "Error Message" in data:
        logger.warning(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
        return pd.DataFrame()

    if "Note" in data:
        logger.warning(f"Alpha Vantage rate limit hit: {data['Note']}")
        return pd.DataFrame()

    ts_key = "Time Series (Daily)"
    if ts_key not in data:
        logger.warning(f"No time series data in Alpha Vantage response for {symbol}")
        return pd.DataFrame()

    rows = []
    for date_str, bar in data[ts_key].items():
        bar_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if bar_date < start_date or bar_date > end_date:
            continue
        rows.append({
            "symbol": symbol,
            "date": bar_date,
            "open": float(bar.get("1. open", 0)),
            "high": float(bar.get("2. high", 0)),
            "low": float(bar.get("3. low", 0)),
            "close": float(bar.get("4. close", 0)),
            "adj_close": float(bar.get("5. adjusted close", 0)),
            "volume": int(float(bar.get("6. volume", 0))),
            "source": "alpha_vantage",
            "event_time": pd.Timestamp(bar_date),
            "as_of_time": datetime.now(UTC),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def reconcile_bars(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    tolerance_pct: float = 2.0,
) -> dict:
    """Cross-check two DataFrames of daily bars for the same symbol.

    Flags discrepancies beyond tolerance_pct between close prices.
    Used to detect yfinance split adjustment errors.

    Returns:
        dict with 'matched', 'mismatched', 'primary_only', 'secondary_only',
        and 'discrepancies' (list of date + both closes).
    """
    if primary_df.empty or secondary_df.empty:
        return {
            "matched": 0,
            "mismatched": 0,
            "primary_only": len(primary_df),
            "secondary_only": len(secondary_df),
            "discrepancies": [],
        }

    p = primary_df.set_index("date")[["close"]].rename(columns={"close": "p_close"})
    s = secondary_df.set_index("date")[["close"]].rename(columns={"close": "s_close"})
    merged = p.join(s, how="outer")

    both = merged.dropna()
    pct_diff = ((both["p_close"] - both["s_close"]) / both["s_close"].replace(0, np.nan)).abs() * 100

    mismatched = pct_diff[pct_diff > tolerance_pct]
    discrepancies = []
    for d in mismatched.index:
        discrepancies.append({
            "date": str(d),
            "primary_close": float(both.loc[d, "p_close"]),
            "secondary_close": float(both.loc[d, "s_close"]),
            "diff_pct": round(float(pct_diff.loc[d]), 2),
        })

    primary_only = merged["s_close"].isna().sum()
    secondary_only = merged["p_close"].isna().sum()

    if discrepancies:
        symbol = primary_df["symbol"].iloc[0] if "symbol" in primary_df.columns else "?"
        logger.warning(
            "%s: %d/%d bars differ >%.1f%% between sources. "
            "Worst: %s",
            symbol, len(discrepancies), len(both), tolerance_pct,
            discrepancies[:3],
        )

    return {
        "matched": int(len(both) - len(mismatched)),
        "mismatched": int(len(mismatched)),
        "primary_only": int(primary_only),
        "secondary_only": int(secondary_only),
        "discrepancies": discrepancies[:20],
    }
