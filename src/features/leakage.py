"""Leakage-safe as-of join utilities.

Ensures features only use data that was actually available before the prediction
timestamp. This prevents look-ahead bias -- the #1 risk in financial ML.
"""

import logging
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def as_of_join_market(
    db: Session,
    symbol: str,
    as_of_date: date,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Load market bars available BEFORE as_of_date (strictly < as_of_date).

    Point-in-time safe: only returns data from sessions before the prediction date.
    """
    result = db.execute(text("""
        SELECT symbol, date, open, high, low, close, adj_close, volume
        FROM market_bars_daily
        WHERE symbol = :symbol
          AND date < :as_of_date
          AND date >= :start_date
        ORDER BY date
    """), {
        "symbol": symbol,
        "as_of_date": as_of_date,
        "start_date": as_of_date - pd.Timedelta(days=lookback_days),
    })

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def as_of_join_macro(
    db: Session,
    series_id: str,
    as_of_date: date,
) -> float | None:
    """Get latest macro value available BEFORE as_of_date.

    Uses available_at_utc to respect release lag -- a CPI number released on
    the 12th shouldn't be used for predictions on the 11th.
    """
    result = db.execute(text("""
        SELECT value FROM macro_series
        WHERE series_id = :series_id
          AND (available_at_utc IS NULL OR available_at_utc::date <= :as_of_date)
          AND observation_date < :as_of_date
        ORDER BY observation_date DESC
        LIMIT 1
    """), {"series_id": series_id, "as_of_date": as_of_date})

    row = result.fetchone()
    return float(row[0]) if row else None


def as_of_join_news_sentiment(
    db: Session,
    symbol: str,
    as_of_date: date,
    lookback_hours: int = 24,
) -> float | None:
    """Get average news sentiment available BEFORE as_of_date.

    Only uses articles with published_time before as_of_date.
    """
    result = db.execute(text("""
        SELECT AVG(ne.sentiment_score)
        FROM news_events ne
        JOIN news_symbol_mapping nsm ON ne.event_id = nsm.event_id
        WHERE nsm.symbol = :symbol
          AND ne.published_time < :as_of_date::timestamp
          AND ne.published_time >= :as_of_date::timestamp - make_interval(hours => :hours)
          AND ne.sentiment_score IS NOT NULL
    """), {
        "symbol": symbol,
        "as_of_date": str(as_of_date),
        "hours": lookback_hours,
    })

    row = result.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def build_training_dataset(
    db: Session,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Build a leakage-safe training dataset from features_snapshot.

    Each feature row is joined with the NEXT session's actual return as the target.
    This ensures we predict tomorrow using only today's features.
    """
    result = db.execute(text("""
        SELECT
            fs.symbol,
            fs.target_session_date,
            fs.rsi_14,
            fs.momentum_5d,
            fs.momentum_10d,
            fs.rolling_return_5d,
            fs.rolling_return_20d,
            fs.rolling_volatility_20d,
            fs.macd,
            fs.macd_signal,
            fs.sector_return_1d,
            fs.sector_return_5d,
            fs.benchmark_relative_return_1d,
            fs.news_sentiment_24h,
            fs.news_sentiment_7d,
            fs.vix_level,
            fs.sp500_momentum_200d,
            fs.regime_label,
            -- Target: next day's return (label)
            LEAD(mb.close, 1) OVER (PARTITION BY fs.symbol ORDER BY fs.target_session_date) / mb.close - 1 AS next_day_return
        FROM features_snapshot fs
        JOIN market_bars_daily mb
            ON fs.symbol = mb.symbol AND fs.target_session_date = mb.date
        WHERE fs.target_session_date >= :start_date
          AND fs.target_session_date <= :end_date
          AND fs.symbol = ANY(:symbols)
        ORDER BY fs.symbol, fs.target_session_date
    """), {"start_date": start_date, "end_date": end_date, "symbols": symbols})

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    columns = [
        "symbol", "target_session_date",
        "rsi_14", "momentum_5d", "momentum_10d",
        "rolling_return_5d", "rolling_return_20d", "rolling_volatility_20d",
        "macd", "macd_signal",
        "sector_return_1d", "sector_return_5d",
        "benchmark_relative_return_1d",
        "news_sentiment_24h", "news_sentiment_7d",
        "vix_level", "sp500_momentum_200d",
        "regime_label",
        "next_day_return",
    ]

    df = pd.DataFrame(rows, columns=columns)
    df["target_session_date"] = pd.to_datetime(df["target_session_date"])

    df["direction"] = (df["next_day_return"] > 0).astype(int)

    df = df.dropna(subset=["next_day_return"])

    logger.info(
        f"Training dataset: {len(df)} rows, "
        f"{df['symbol'].nunique()} symbols, "
        f"{df['target_session_date'].min()} to {df['target_session_date'].max()}"
    )
    return df


def validate_no_leakage(df: pd.DataFrame) -> dict:
    """Run leakage checks on a training dataset.

    Returns dict with check results and any violations found.
    """
    checks = {"passed": True, "violations": []}

    for symbol in df["symbol"].unique():
        sym_df = df[df["symbol"] == symbol].sort_values("target_session_date")
        dates = sym_df["target_session_date"].values

        for i in range(1, len(dates)):
            if dates[i] <= dates[i - 1]:
                checks["violations"].append(
                    f"{symbol}: non-monotonic dates at index {i}"
                )
                checks["passed"] = False

    if df["next_day_return"].isna().sum() > 0:
        na_count = df["next_day_return"].isna().sum()
        checks["violations"].append(f"{na_count} rows with NaN target (expected for last day)")

    logger.info(f"Leakage check: {'PASSED' if checks['passed'] else 'FAILED'} ({len(checks['violations'])} issues)")
    return checks
