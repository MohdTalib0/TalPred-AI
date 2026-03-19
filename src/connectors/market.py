"""Market data connector using yfinance.

Fetches daily OHLCV data for symbols and returns validated DataFrames
ready for ingestion into market_bars_daily.

Hardened with:
  - Exponential backoff retry (3 attempts) for transient API failures
  - Split/dividend adjustment ratio validation
  - Stale data detection (warns if latest bar is >3 trading days old)
  - Price continuity checks (flags >50% single-day moves as potential
    bad splits)
"""

import logging
import time
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 2.0
_STALE_THRESHOLD_DAYS = 5
_SPLIT_JUMP_THRESHOLD = 0.50


def fetch_daily_bars(
    symbol: str,
    start_date: date,
    end_date: date,
    max_retries: int = _MAX_RETRIES,
) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance with retry and validation.

    Returns DataFrame with columns:
        symbol, date, open, high, low, close, adj_close, volume,
        source, event_time, as_of_time
    """
    df = _fetch_with_retry(symbol, start_date, end_date, max_retries)

    if df.empty:
        logger.warning(f"No data returned for {symbol} from {start_date} to {end_date}")
        return pd.DataFrame()

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })

    df["symbol"] = symbol
    df["source"] = "yfinance"
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["event_time"] = pd.to_datetime(df["date"])
    df["as_of_time"] = datetime.now(UTC)

    df = _validate_adjustment_ratio(df, symbol)
    df = _flag_price_jumps(df, symbol)
    _check_staleness(df, symbol, end_date)

    columns = [
        "symbol", "date", "open", "high", "low", "close",
        "adj_close", "volume", "source", "event_time", "as_of_time",
    ]
    available = [c for c in columns if c in df.columns]
    return df[available]


def fetch_multiple_symbols(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch daily bars for multiple symbols. Returns combined DataFrame."""
    frames = []
    for symbol in symbols:
        try:
            df = fetch_daily_bars(symbol, start_date, end_date)
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.exception(f"Failed to fetch {symbol}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _fetch_with_retry(
    symbol: str,
    start_date: date,
    end_date: date,
    max_retries: int,
) -> pd.DataFrame:
    """Fetch with exponential backoff on transient yfinance failures."""
    ticker = yf.Ticker(symbol)
    last_err = None

    for attempt in range(max_retries):
        try:
            df = ticker.history(
                start=start_date, end=end_date,
                auto_adjust=False, repair=True,
            )
            if df is not None:
                return df
        except Exception as e:
            last_err = e
            wait = _RETRY_BASE_SECONDS * (2 ** attempt)
            logger.warning(
                "%s: fetch attempt %d/%d failed (%s), retrying in %.1fs",
                symbol, attempt + 1, max_retries, e, wait,
            )
            time.sleep(wait)

    logger.error("%s: all %d fetch attempts failed: %s", symbol, max_retries, last_err)
    return pd.DataFrame()


def _validate_adjustment_ratio(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Check that adj_close / close ratio is consistent across the series.

    A sudden ratio change without a corresponding stock split event
    suggests yfinance retroactively adjusted historical data mid-series.
    """
    if "adj_close" not in df.columns or "close" not in df.columns:
        return df
    if len(df) < 2:
        return df

    ratio = df["adj_close"] / df["close"].replace(0, np.nan)
    ratio = ratio.dropna()
    if len(ratio) < 2:
        return df

    ratio_diff = ratio.diff().abs()
    large_shifts = ratio_diff[ratio_diff > 0.01]

    if len(large_shifts) > 3:
        logger.warning(
            "%s: %d adjustment ratio shifts detected — possible retroactive "
            "split adjustment error in yfinance data",
            symbol, len(large_shifts),
        )

    return df


def _flag_price_jumps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flag single-day price moves >50% as potential bad split data."""
    if "close" not in df.columns or len(df) < 2:
        return df

    returns = df["close"].pct_change().abs()
    jumps = returns[returns > _SPLIT_JUMP_THRESHOLD]

    if len(jumps) > 0:
        jump_dates = df.loc[jumps.index, "date"].tolist()
        logger.warning(
            "%s: %d suspicious price jumps >%.0f%% on %s — "
            "may indicate unadjusted split. Verify manually.",
            symbol, len(jumps), _SPLIT_JUMP_THRESHOLD * 100,
            jump_dates[:5],
        )

    return df


def _check_staleness(
    df: pd.DataFrame,
    symbol: str,
    requested_end: date,
) -> None:
    """Warn if the latest available bar is significantly older than requested."""
    if df.empty or "date" not in df.columns:
        return

    latest_date = max(df["date"])
    if isinstance(latest_date, datetime):
        latest_date = latest_date.date()

    gap_days = (requested_end - latest_date).days
    if gap_days > _STALE_THRESHOLD_DAYS:
        logger.warning(
            "%s: stale data — latest bar is %s (%d days before requested end %s). "
            "Stock may be delisted or yfinance may be throttling.",
            symbol, latest_date, gap_days, requested_end,
        )
