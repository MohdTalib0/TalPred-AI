"""Market data connector using yfinance.

Fetches daily OHLCV data for symbols and returns validated DataFrames
ready for ingestion into market_bars_daily.
"""

import logging
from datetime import UTC, date, datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_daily_bars(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance.

    Returns DataFrame with columns:
        symbol, date, open, high, low, close, adj_close, volume, source, event_time, as_of_time
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

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

    columns = [
        "symbol", "date", "open", "high", "low", "close",
        "adj_close", "volume", "source", "event_time", "as_of_time",
    ]
    return df[columns]


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
