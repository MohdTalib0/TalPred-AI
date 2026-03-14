"""Feature generation engine.

Supports two modes:
  - backfill: compute features for all historical dates
  - incremental: compute features only for the latest session (daily use)

Features computed (wide SQL columns matching features_snapshot schema):
  Technical:  rsi_14, momentum_5d, momentum_10d, momentum_20d, momentum_60d,
              momentum_120d, rolling_return_5d, rolling_return_20d,
              rolling_volatility_20d, macd, macd_signal, short_term_reversal,
              volume_change_5d, volume_zscore_20d, volatility_expansion_5_20
  Flow:       volume_acceleration, signed_volume_proxy, price_volume_trend,
              volume_imbalance_proxy, liquidity_shock_5d, vwap_deviation
  Sector:     sector_return_1d, sector_return_5d,
              sector_relative_return_1d, sector_relative_return_5d
  Cross-sec:  momentum_rank_market, momentum_60d_rank_market,
              momentum_120d_rank_market, short_term_reversal_rank_market,
              volatility_rank_market, rsi_rank_market, volume_rank_market,
              sector_momentum_rank
  Benchmark:  benchmark_relative_return_1d
  Macro:      vix_level, sp500_momentum_200d
  Regime:     regime_label
  NLP:        news_sentiment_24h, news_sentiment_7d,
              news_sentiment_std, news_positive_ratio, news_negative_ratio,
              news_volume, news_credibility_avg, news_present_flag
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
    """Load credibility-weighted sentiment features aggregated by symbol and date.

    Sentiment is weighted by source credibility so that high-quality sources
    (Reuters, CNBC) influence the aggregate more than low-quality ones (Yahoo).
    """
    result = db.execute(text("""
        SELECT
            nsm.symbol,
            ne.published_time::date AS pub_date,
            ne.sentiment_score,
            COALESCE(ne.credibility_score, 0.4) AS credibility
        FROM news_events ne
        JOIN news_symbol_mapping nsm ON ne.event_id = nsm.event_id
        WHERE ne.sentiment_score IS NOT NULL
        ORDER BY nsm.symbol, pub_date
    """))
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "date", "sentiment_24h", "sentiment_7d",
            "sentiment_std", "positive_ratio", "negative_ratio",
            "news_volume", "credibility_avg",
        ])

    raw = pd.DataFrame(rows, columns=["symbol", "pub_date", "sentiment_score", "credibility"])

    records = []
    for symbol, grp in raw.groupby("symbol"):
        daily_groups = grp.groupby("pub_date")
        daily_rows = []
        for pub_date, day_grp in daily_groups:
            sents = day_grp["sentiment_score"].values
            creds = day_grp["credibility"].values
            weight_total = creds.sum()

            weighted_mean = (sents * creds).sum() / weight_total if weight_total > 0 else sents.mean()
            n = len(sents)
            daily_rows.append({
                "pub_date": pub_date,
                "weighted_sent": float(weighted_mean),
                "std_sent": float(sents.std()) if n > 1 else 0.0,
                "count": n,
                "positive": int((sents > 0.1).sum()),
                "negative": int((sents < -0.1).sum()),
                "cred_avg": float(creds.mean()),
            })

        daily = pd.DataFrame(daily_rows).sort_values("pub_date")

        for _, row in daily.iterrows():
            d = row["pub_date"]
            week_ago = d - pd.Timedelta(days=7)
            week_rows = daily[(daily["pub_date"] >= week_ago) & (daily["pub_date"] <= d)]
            sent_7d = week_rows["weighted_sent"].mean() if len(week_rows) > 0 else row["weighted_sent"]

            records.append({
                "symbol": symbol,
                "date": d,
                "sentiment_24h": row["weighted_sent"],
                "sentiment_7d": float(sent_7d),
                "sentiment_std": row["std_sent"],
                "positive_ratio": row["positive"] / row["count"] if row["count"] > 0 else 0.0,
                "negative_ratio": row["negative"] / row["count"] if row["count"] > 0 else 0.0,
                "news_volume": float(row["count"]),
                "credibility_avg": row["cred_avg"],
            })

    return pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "symbol", "date", "sentiment_24h", "sentiment_7d",
        "sentiment_std", "positive_ratio", "negative_ratio",
        "news_volume", "credibility_avg",
    ])


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


def load_vix_series(db: Session, lookback_days: int = 1500) -> pd.Series:
    """Load full VIX time series for point-in-time lookup."""
    result = db.execute(text("""
        SELECT observation_date, value FROM macro_series
        WHERE series_id = 'VIXCLS'
          AND observation_date >= CURRENT_DATE - :lookback
        ORDER BY observation_date
    """), {"lookback": lookback_days})
    rows = result.fetchall()
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"]


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
    df["momentum_20d"] = close.pct_change(20)
    df["momentum_60d"] = close.pct_change(60)
    df["momentum_120d"] = close.pct_change(120)
    df["rolling_return_5d"] = close.pct_change(5)
    df["rolling_return_20d"] = close.pct_change(20)
    rets = close.pct_change()
    df["short_term_reversal"] = -rets
    df["rolling_volatility_20d"] = rets.rolling(20).std()
    df["rolling_volatility_5d"] = rets.rolling(5).std()
    df["stock_return_1d"] = rets
    df["volume_change_5d"] = df["volume"].pct_change(5)
    vol_mean_20 = df["volume"].rolling(20).mean()
    vol_std_20 = df["volume"].rolling(20).std()
    df["volume_zscore_20d"] = (df["volume"] - vol_mean_20) / vol_std_20.replace(0, np.nan)
    df["volatility_expansion_5_20"] = df["rolling_volatility_5d"] / df["rolling_volatility_20d"].replace(0, np.nan)

    macd_line, signal_line = _macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line

    # --- Flow-based liquidity signals ---
    vol_change_20 = df["volume"].pct_change(20)
    # Acceleration: 5-day volume trend minus 20-day volume trend (sudden surges vs recent baseline)
    df["volume_acceleration"] = df["volume_change_5d"] - vol_change_20

    # Signed order-flow proxy: direction of return × magnitude of volume shock.
    # Positive → price rose on high relative volume (buying pressure).
    # Negative → price fell on high relative volume (selling pressure).
    df["signed_volume_proxy"] = np.sign(rets) * df["volume_zscore_20d"]

    # Price-volume trend: co-movement of 5-day price momentum and 5-day volume change.
    # Positive → confirmed trend (price + volume both moving in same direction).
    # Negative → divergence (price up but volume falling, or vice versa).
    df["price_volume_trend"] = df["momentum_5d"] * df["volume_change_5d"].clip(-2.0, 2.0)

    # --- OHLCV bar-shape flow signals (require high/low data) ---

    # Volume imbalance proxy: signed volume weighted by intrabar directional bias.
    # Uses candle body as a proxy for intraday order-flow direction:
    #   +1 = pure bullish candle (close == high, open == low)
    #   -1 = pure bearish candle
    # Multiplied by volume_zscore to capture direction × shock magnitude.
    # Clip to [-5, 5] to prevent near-zero hl_range producing extreme values.
    hl_range = (df["high"] - df["low"]).where(lambda x: x > 1e-6, np.nan)
    candle_body_ratio = ((close - df["open"]) / hl_range).clip(-1.0, 1.0)
    df["volume_imbalance_proxy"] = (candle_body_ratio * df["volume_zscore_20d"]).clip(-10.0, 10.0)

    # Liquidity shock 5d: max absolute volume z-score over the last 5 bars.
    # Captures "was there any extreme volume event (buy OR sell) in the past week?"
    # High values indicate institutional activity that may persist 1-5 days.
    df["liquidity_shock_5d"] = df["volume_zscore_20d"].abs().rolling(5, min_periods=2).max().clip(0, 15)

    # VWAP deviation: how far did the close land from the day's typical price.
    # Typical price ≈ (high + low + close) / 3
    # Positive → closed near the high (buying pressure dominated the session).
    # Negative → closed near the low (selling pressure dominated).
    typical_price = (df["high"] + df["low"] + close) / 3
    df["vwap_deviation"] = ((close - typical_price) / typical_price.where(lambda x: x > 1e-6, np.nan)).clip(-0.5, 0.5)

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
    sp500_series = load_sp500_data(db, lookback_days=lb)
    vix_series = load_vix_series(db, lookback_days=lb)

    sp500_dates_norm = sp500_series.index.normalize() if len(sp500_series) > 0 else pd.DatetimeIndex([])

    sector_returns_df = compute_sector_returns(market_df, sector_map)

    # Pre-compute technical features for all symbols, then derive cross-sectional ranks by date.
    all_symbol_features: list[pd.DataFrame] = []
    for symbol in symbols:
        sym_df = market_df[market_df["symbol"] == symbol].copy().sort_values("date")
        if sym_df.empty:
            continue
        sym_df = compute_technical_features(sym_df)
        sym_df["sector"] = sector_map.get(symbol)
        all_symbol_features.append(sym_df)
    all_feat_df = pd.concat(all_symbol_features, ignore_index=True) if all_symbol_features else pd.DataFrame()
    if not all_feat_df.empty:
        all_feat_df["momentum_rank_market"] = all_feat_df.groupby("date")["momentum_5d"].rank(pct=True)
        all_feat_df["momentum_60d_rank_market"] = all_feat_df.groupby("date")["momentum_60d"].rank(pct=True)
        all_feat_df["momentum_120d_rank_market"] = all_feat_df.groupby("date")["momentum_120d"].rank(pct=True)
        all_feat_df["short_term_reversal_rank_market"] = all_feat_df.groupby("date")["short_term_reversal"].rank(pct=True)
        all_feat_df["volatility_rank_market"] = all_feat_df.groupby("date")["rolling_volatility_20d"].rank(pct=True)
        all_feat_df["rsi_rank_market"] = all_feat_df.groupby("date")["rsi_14"].rank(pct=True)
        all_feat_df["volume_rank_market"] = all_feat_df.groupby("date")["volume"].rank(pct=True)
        all_feat_df["sector_momentum_rank"] = all_feat_df.groupby(["date", "sector"])["momentum_5d"].rank(pct=True)
        all_feat_df = all_feat_df.sort_values(["symbol", "date"])
    now = datetime.now(UTC)

    snapshots = []
    for symbol in symbols:
        sym_df = all_feat_df[all_feat_df["symbol"] == symbol].copy().sort_values("date") if not all_feat_df.empty else pd.DataFrame()
        if len(sym_df) < 30:
            continue

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
            sector_relative_1d = None
            if _to_float(row.get("stock_return_1d")) is not None and sector_1d is not None:
                sector_relative_1d = _to_float(row.get("stock_return_1d") - sector_1d)
            sector_relative_5d = None
            if _to_float(row.get("rolling_return_5d")) is not None and sector_5d is not None:
                sector_relative_5d = _to_float(row.get("rolling_return_5d") - sector_5d)

            # Point-in-time VIX: latest VIX on or before target date
            vix_val = None
            if len(vix_series) > 0:
                vix_before = vix_series[vix_series.index.normalize() <= pd.Timestamp(td)]
                if len(vix_before) > 0:
                    vix_val = _to_float(vix_before.iloc[-1])

            # Point-in-time SP500 momentum: 200-day return ending at target date
            sp500_mom = None
            benchmark_rel = None
            if len(sp500_series) > 1:
                sp_before = sp500_series[sp500_dates_norm <= pd.Timestamp(td)]
                if len(sp_before) >= 200:
                    sp500_mom = _to_float((sp_before.iloc[-1] / sp_before.iloc[-200]) - 1)

                if len(sp_before) >= 2:
                    sp_ret = (sp_before.iloc[-1] / sp_before.iloc[-2]) - 1

                    row_idx = sym_df.index[sym_df["date"].dt.date == td]
                    if len(row_idx) > 0:
                        row_loc = sym_df.index.get_loc(row_idx[0])
                        if row_loc > 0:
                            stock_ret = (sym_df["close"].iloc[row_loc] / sym_df["close"].iloc[row_loc - 1]) - 1
                            benchmark_rel = _to_float(stock_ret - sp_ret)

            regime = classify_regime(sp500_mom, vix_val)

            news_sent_24h = _lookup_sentiment(sentiment_df, symbol, td, "sentiment_24h")
            news_present_flag = 1.0 if news_sent_24h is not None else 0.0

            snapshots.append({
                "snapshot_id": _snapshot_id(symbol, td),
                "symbol": symbol,
                "as_of_time": now,
                "target_session_date": td,
                "rsi_14": _to_float(row.get("rsi_14")),
                "momentum_5d": _to_float(row.get("momentum_5d")),
                "momentum_10d": _to_float(row.get("momentum_10d")),
                "momentum_20d": _to_float(row.get("momentum_20d")),
                "momentum_60d": _to_float(row.get("momentum_60d")),
                "momentum_120d": _to_float(row.get("momentum_120d")),
                "rolling_return_5d": _to_float(row.get("rolling_return_5d")),
                "rolling_return_20d": _to_float(row.get("rolling_return_20d")),
                "rolling_volatility_20d": _to_float(row.get("rolling_volatility_20d")),
                "macd": _to_float(row.get("macd")),
                "macd_signal": _to_float(row.get("macd_signal")),
                "short_term_reversal": _to_float(row.get("short_term_reversal")),
                "sector_return_1d": sector_1d,
                "sector_return_5d": sector_5d,
                "sector_relative_return_1d": sector_relative_1d,
                "sector_relative_return_5d": sector_relative_5d,
                "momentum_rank_market": _to_float(row.get("momentum_rank_market")),
                "momentum_60d_rank_market": _to_float(row.get("momentum_60d_rank_market")),
                "momentum_120d_rank_market": _to_float(row.get("momentum_120d_rank_market")),
                "short_term_reversal_rank_market": _to_float(row.get("short_term_reversal_rank_market")),
                "volatility_rank_market": _to_float(row.get("volatility_rank_market")),
                "rsi_rank_market": _to_float(row.get("rsi_rank_market")),
                "volume_rank_market": _to_float(row.get("volume_rank_market")),
                "sector_momentum_rank": _to_float(row.get("sector_momentum_rank")),
                "volume_change_5d": _to_float(row.get("volume_change_5d")),
                "volume_zscore_20d": _to_float(row.get("volume_zscore_20d")),
                "volatility_expansion_5_20": _to_float(row.get("volatility_expansion_5_20")),
                "volume_acceleration": _to_float(row.get("volume_acceleration")),
                "signed_volume_proxy": _to_float(row.get("signed_volume_proxy")),
                "price_volume_trend": _to_float(row.get("price_volume_trend")),
                "volume_imbalance_proxy": _to_float(row.get("volume_imbalance_proxy")),
                "liquidity_shock_5d": _to_float(row.get("liquidity_shock_5d")),
                "vwap_deviation": _to_float(row.get("vwap_deviation")),
                "benchmark_relative_return_1d": benchmark_rel,
                "news_sentiment_24h": news_sent_24h,
                "news_sentiment_7d": _lookup_sentiment(sentiment_df, symbol, td, "sentiment_7d"),
                "news_sentiment_std": _lookup_sentiment(sentiment_df, symbol, td, "sentiment_std"),
                "news_positive_ratio": _lookup_sentiment(sentiment_df, symbol, td, "positive_ratio"),
                "news_negative_ratio": _lookup_sentiment(sentiment_df, symbol, td, "negative_ratio"),
                "news_volume": _lookup_sentiment(sentiment_df, symbol, td, "news_volume"),
                "news_credibility_avg": _lookup_sentiment(sentiment_df, symbol, td, "credibility_avg"),
                "news_present_flag": news_present_flag,
                "vix_level": vix_val,
                "sp500_momentum_200d": sp500_mom,
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
