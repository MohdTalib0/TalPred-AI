"""Feature generation engine.

Supports two modes:
  - backfill: compute features for all historical dates
  - incremental: compute features only for the latest session (daily use)

Features computed (wide SQL columns matching features_snapshot schema):
  Technical: rsi_14, momentum_5d, momentum_10d, rolling_return_5d,
             rolling_return_20d, rolling_volatility_20d, macd, macd_signal
  Sector:    sector_return_1d, sector_return_5d
  Benchmark: benchmark_relative_return_1d
  Macro:     vix_level, sp500_momentum_200d
  Regime:    regime_label
  NLP:       news_sentiment_24h, news_sentiment_7d (placeholder for FinBERT)
"""

import hashlib
import logging
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


def _lookup_sentiment(sentiment_df: pd.DataFrame, symbol: str, td: date, col: str) -> float | None:
    """Look up sentiment value for a symbol on a specific date."""
    if sentiment_df.empty:
        return None
    match = sentiment_df[(sentiment_df["symbol"] == symbol) & (sentiment_df["date"] == td)]
    if match.empty:
        return None
    return _to_float(match.iloc[0][col])


def _to_float(val) -> float | None:
    """Convert numpy/pandas numeric types to native Python float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return float(val)

logger = logging.getLogger(__name__)

MAX_LOOKBACK_DAYS = 310  # ~200 trading days + buffer for weekends/holidays


def load_sentiment_data(db: Session) -> pd.DataFrame:
    """Load pre-computed sentiment scores from news_events aggregated by symbol and date.

    Returns DataFrame with columns: symbol, date, sentiment_24h, sentiment_7d
    """
    result = db.execute(text("""
        SELECT
            nsm.symbol,
            ne.published_time::date AS pub_date,
            AVG(ne.sentiment_score) AS avg_sentiment
        FROM news_events ne
        JOIN news_symbol_mapping nsm ON ne.event_id = nsm.event_id
        WHERE ne.sentiment_score IS NOT NULL
        GROUP BY nsm.symbol, ne.published_time::date
        ORDER BY nsm.symbol, pub_date
    """))
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "sentiment_24h", "sentiment_7d"])

    df = pd.DataFrame(rows, columns=["symbol", "date", "avg_sentiment"])

    records = []
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("date")
        for _, row in grp.iterrows():
            d = row["date"]
            sent_24h = row["avg_sentiment"]
            week_ago = d - pd.Timedelta(days=7)
            sent_7d_rows = grp[(grp["date"] >= week_ago) & (grp["date"] <= d)]
            sent_7d = sent_7d_rows["avg_sentiment"].mean() if len(sent_7d_rows) > 0 else sent_24h
            records.append({
                "symbol": symbol,
                "date": d,
                "sentiment_24h": float(sent_24h),
                "sentiment_7d": float(sent_7d),
            })

    return pd.DataFrame(records) if records else pd.DataFrame(columns=["symbol", "date", "sentiment_24h", "sentiment_7d"])


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return macd_line, signal_line


def _snapshot_id(symbol: str, target_date: date) -> str:
    raw = f"{symbol}:{target_date.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


def load_market_window(db: Session, symbols: list[str], lookback_days: int = MAX_LOOKBACK_DAYS) -> pd.DataFrame:
    """Load recent market bars for all symbols in one query."""
    query = text("""
        SELECT symbol, date, open, high, low, close, adj_close, volume
        FROM market_bars_daily
        WHERE symbol = ANY(:symbols)
          AND date >= CURRENT_DATE - :lookback
        ORDER BY symbol, date
    """)
    result = db.execute(query, {"symbols": symbols, "lookback": lookback_days})
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_sector_map(db: Session) -> dict[str, str]:
    """Load symbol → sector mapping."""
    result = db.execute(text("SELECT symbol, sector FROM symbols WHERE is_active = true"))
    return {row[0]: row[1] for row in result.fetchall()}


def load_macro_latest(db: Session) -> dict:
    """Load latest VIX value."""
    result = db.execute(text("""
        SELECT value FROM macro_series
        WHERE series_id = 'VIXCLS'
        ORDER BY observation_date DESC LIMIT 1
    """))
    row = result.fetchone()
    return {"vix": float(row[0]) if row else None}


def load_sp500_data(db: Session, lookback_days: int = MAX_LOOKBACK_DAYS) -> pd.Series:
    """Load SPY (S&P 500 ETF) as benchmark."""
    result = db.execute(text("""
        SELECT date, close FROM market_bars_daily
        WHERE symbol = 'SPY'
        AND date >= CURRENT_DATE - :lookback
        ORDER BY date
    """), {"lookback": lookback_days})
    rows = result.fetchall()
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"]


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol technical features. Input must be sorted by date."""
    close = df["close"]

    df = df.copy()
    df["rsi_14"] = _rsi(close, 14)
    df["momentum_5d"] = close.pct_change(5)
    df["momentum_10d"] = close.pct_change(10)
    df["rolling_return_5d"] = close.pct_change(5)
    df["rolling_return_20d"] = close.pct_change(20)
    df["rolling_volatility_20d"] = close.pct_change().rolling(20).std()

    macd_line, signal_line = _macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line

    return df


def compute_sector_returns(market_df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """Compute sector-level daily returns. Returns a DataFrame with sector, date, returns."""
    market_df = market_df.copy()
    market_df["sector"] = market_df["symbol"].map(sector_map)
    market_df = market_df.dropna(subset=["sector"])
    market_df["daily_return"] = market_df.groupby("symbol")["close"].pct_change()

    sector_returns = (
        market_df.groupby(["sector", "date"])["daily_return"]
        .mean()
        .reset_index()
        .rename(columns={"daily_return": "sector_return_1d"})
    )

    sector_returns = sector_returns.sort_values(["sector", "date"])
    sector_returns["sector_return_5d"] = (
        sector_returns.groupby("sector")["sector_return_1d"]
        .transform(lambda x: x.rolling(5).sum())
    )

    return sector_returns


def classify_regime(sp500_momentum_200d: float | None, vix: float | None, sideways_pct: float = 0.02, vix_threshold: float = 20) -> str:
    if sp500_momentum_200d is None or vix is None:
        return "unknown"

    is_bull = sp500_momentum_200d > sideways_pct
    is_bear = sp500_momentum_200d < -sideways_pct
    is_high_vol = vix > vix_threshold

    if is_bull and not is_high_vol:
        return "bull_low_vol"
    elif is_bull and is_high_vol:
        return "bull_high_vol"
    elif is_bear and is_high_vol:
        return "bear_high_vol"
    elif is_bear and not is_high_vol:
        return "bear_low_vol"
    else:
        return "sideways"


def generate_features(
    db: Session,
    symbols: list[str],
    target_dates: list[date] | None = None,
    dataset_version: str | None = None,
    lookback_days: int | None = None,
) -> list[dict]:
    """Generate feature snapshots for given symbols.

    If target_dates is None, computes for the latest available date only (incremental mode).
    If target_dates is provided, computes for all those dates (backfill mode).
    """
    lb = lookback_days or MAX_LOOKBACK_DAYS
    logger.info(f"Loading market window for {len(symbols)} symbols (lookback={lb})...")
    market_df = load_market_window(db, symbols, lookback_days=lb)
    sentiment_df = load_sentiment_data(db)
    if market_df.empty:
        logger.warning("No market data found")
        return []

    sector_map = load_sector_map(db)
    macro = load_macro_latest(db)
    sp500_series = load_sp500_data(db)

    sp500_momentum_200d = None
    if len(sp500_series) >= 200:
        sp500_momentum_200d = (sp500_series.iloc[-1] / sp500_series.iloc[-200]) - 1

    regime = classify_regime(sp500_momentum_200d, macro.get("vix"))
    logger.info(f"Current regime: {regime}, VIX: {macro.get('vix')}, SP500 200d: {sp500_momentum_200d}")

    sector_returns_df = compute_sector_returns(market_df, sector_map)
    now = datetime.now(UTC)

    snapshots = []
    for symbol in symbols:
        sym_df = market_df[market_df["symbol"] == symbol].copy().sort_values("date")
        if len(sym_df) < 30:
            continue

        sym_df = compute_technical_features(sym_df)

        sector = sector_map.get(symbol)
        sym_sector_returns = sector_returns_df[sector_returns_df["sector"] == sector] if sector else pd.DataFrame()

        if target_dates is None:
            dates_to_process = [sym_df["date"].iloc[-1].date()]
        else:
            available = set(sym_df["date"].dt.date)
            dates_to_process = [d for d in target_dates if d in available]

        for td in dates_to_process:
            row = sym_df[sym_df["date"].dt.date == td]
            if row.empty:
                continue
            row = row.iloc[-1]

            sector_1d = None
            sector_5d = None
            if not sym_sector_returns.empty:
                sr = sym_sector_returns[sym_sector_returns["date"].dt.date == td] if "date" in sym_sector_returns.columns else pd.DataFrame()
                if not sr.empty:
                    sector_1d = _to_float(sr.iloc[0]["sector_return_1d"])
                    sector_5d = _to_float(sr.iloc[0]["sector_return_5d"])

            benchmark_rel = None
            if len(sp500_series) > 1:
                ts = pd.Timestamp(td)
                if ts in sp500_series.index and sp500_series.index.get_loc(ts) > 0:
                    loc = sp500_series.index.get_loc(ts)
                    sp_ret = (sp500_series.iloc[loc] / sp500_series.iloc[loc - 1]) - 1

                    row_idx = sym_df.index[sym_df["date"].dt.date == td]
                    row_loc = sym_df.index.get_loc(row_idx[0])
                    if row_loc > 0:
                        stock_ret = (sym_df["close"].iloc[row_loc] / sym_df["close"].iloc[row_loc - 1]) - 1
                        benchmark_rel = _to_float(stock_ret - sp_ret)

            snapshots.append({
                "snapshot_id": _snapshot_id(symbol, td),
                "symbol": symbol,
                "as_of_time": now,
                "target_session_date": td,
                "rsi_14": _to_float(row.get("rsi_14")),
                "momentum_5d": _to_float(row.get("momentum_5d")),
                "momentum_10d": _to_float(row.get("momentum_10d")),
                "rolling_return_5d": _to_float(row.get("rolling_return_5d")),
                "rolling_return_20d": _to_float(row.get("rolling_return_20d")),
                "rolling_volatility_20d": _to_float(row.get("rolling_volatility_20d")),
                "macd": _to_float(row.get("macd")),
                "macd_signal": _to_float(row.get("macd_signal")),
                "sector_return_1d": sector_1d,
                "sector_return_5d": sector_5d,
                "benchmark_relative_return_1d": benchmark_rel,
                "news_sentiment_24h": _lookup_sentiment(sentiment_df, symbol, td, "sentiment_24h"),
                "news_sentiment_7d": _lookup_sentiment(sentiment_df, symbol, td, "sentiment_7d"),
                "vix_level": _to_float(macro.get("vix")),
                "sp500_momentum_200d": _to_float(sp500_momentum_200d),
                "regime_label": regime,
                "dataset_version": dataset_version,
            })

    logger.info(f"Generated {len(snapshots)} feature snapshots")
    return snapshots


def save_snapshots(db: Session, snapshots: list[dict]) -> int:
    """Upsert feature snapshots into DB."""
    if not snapshots:
        return 0

    from src.models.schema import FeaturesSnapshot

    count = 0
    for snap in snapshots:
        existing = db.query(FeaturesSnapshot).filter(
            FeaturesSnapshot.snapshot_id == snap["snapshot_id"]
        ).first()

        if existing:
            for k, v in snap.items():
                setattr(existing, k, v)
        else:
            db.add(FeaturesSnapshot(**snap))
        count += 1

    db.commit()
    logger.info(f"Saved {count} feature snapshots")
    return count


def save_sector_returns(db: Session, sector_returns_df: pd.DataFrame) -> int:
    """Upsert sector returns into sector_returns_daily."""
    if sector_returns_df.empty:
        return 0

    count = 0
    for _, row in sector_returns_df.iterrows():
        db.execute(text("""
            INSERT INTO sector_returns_daily (sector, date, sector_return_1d, sector_return_5d)
            VALUES (:sector, :date, :ret_1d, :ret_5d)
            ON CONFLICT (sector, date) DO UPDATE SET
                sector_return_1d = EXCLUDED.sector_return_1d,
                sector_return_5d = EXCLUDED.sector_return_5d
        """), {
            "sector": row["sector"],
            "date": row["date"],
            "ret_1d": float(row["sector_return_1d"]) if pd.notna(row["sector_return_1d"]) else None,
            "ret_5d": float(row["sector_return_5d"]) if pd.notna(row["sector_return_5d"]) else None,
        })
        count += 1

    db.commit()
    logger.info(f"Saved {count} sector return rows")
    return count
