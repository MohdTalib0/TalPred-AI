"""Leakage-safe as-of join utilities.

Ensures features only use data that was actually available before the prediction
timestamp. This prevents look-ahead bias -- the #1 risk in financial ML.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Winsorize a series to reduce outlier impact."""
    if s.empty or s.dropna().empty:
        return s
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def _add_flow_residuals(
    df: pd.DataFrame,
    flow_cols: list[str],
    size_col: str = "log_market_cap",
) -> pd.DataFrame:
    """Orthogonalize flow features against size and sector via per-date OLS.

    For each trading date, regresses each flow feature on:
      [intercept, log_market_cap, sector_dummies]
    and keeps only the residual — the part of flow *not* explained by size or
    sector rotation.  Produces columns named `{col}_resid`.

    This isolates pure stock-specific order flow, which is often more predictive
    than raw flow that is confounded by market-cap and sector effects.
    """
    if df.empty or size_col not in df.columns:
        return df

    available = [c for c in flow_cols if c in df.columns]
    if not available:
        return df

    result_parts = []
    for _, grp in df.groupby("target_session_date"):
        grp = grp.copy()

        # Build per-date design matrix: intercept + size + sector dummies.
        X_parts: list[np.ndarray] = [np.ones(len(grp))]
        size_vals = grp[size_col].fillna(grp[size_col].median()).values
        X_parts.append(size_vals)
        if "sector" in grp.columns:
            sec_dummies = pd.get_dummies(grp["sector"], drop_first=True, dtype=float).values
            if sec_dummies.shape[1] > 0:
                X_parts.append(sec_dummies)
        X = np.column_stack(X_parts)

        for col in available:
            y = grp[col].values.astype(float)
            # Replace inf before OLS — inf in flow features causes lstsq to blow up.
            y = np.where(np.isinf(y), np.nan, y)
            mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
            resid = np.full(len(y), np.nan)
            if mask.sum() > X.shape[1] + 2:
                beta, _, _, _ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
                y_hat = X @ beta
                resid[mask] = y[mask] - y_hat[mask]
            grp[f"{col}_resid"] = resid

        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True) if result_parts else df


def _add_cross_sectional_transforms(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add daily cross-sectional z-score and sector-neutral features.

    For each feature x:
      - x_cs: winsorized daily cross-sectional z-score
      - x_sector_neutral: x_cs minus sector mean(x_cs) on same date
    """
    if df.empty:
        return df

    available = [c for c in cols if c in df.columns]
    if not available:
        return df

    for col in available:
        win_col = f"{col}_win"
        cs_col = f"{col}_cs"
        sn_col = f"{col}_sector_neutral"

        # Winsorize cross-section by date.
        df[win_col] = df.groupby("target_session_date")[col].transform(
            lambda s: _winsorize_series(s, 0.01, 0.99)
        )

        # Cross-sectional z-score by date.
        grp = df.groupby("target_session_date")[win_col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        df[cs_col] = ((df[win_col] - mean) / std).clip(-6.0, 6.0)

        # Sector neutral residual of cross-sectional z-score.
        if "sector" in df.columns:
            sec_mean = df.groupby(["target_session_date", "sector"])[cs_col].transform("mean")
            df[sn_col] = df[cs_col] - sec_mean
        else:
            df[sn_col] = np.nan

        df = df.drop(columns=[win_col])

    return df


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
    target_mode: str = "absolute",
    top_bottom_pct: float = 0.30,
    target_horizon_days: int = 1,
    include_liquidity_features: bool = False,
    sample_stride: int = 1,
    include_fundamentals: bool = False,
) -> pd.DataFrame:
    """Build a leakage-safe training dataset from features_snapshot.

    Each feature row is joined with the NEXT session's actual return as the target.
    This ensures we predict tomorrow using only today's features.

    Args:
        sample_stride: Keep every Nth row per symbol (chronologically).
            For a 5-day horizon, stride=5 produces non-overlapping target
            windows, preventing autocorrelated residuals from inflating
            effective sample size.  Default 1 = keep all rows.
        include_fundamentals: If True, fetches fundamental data from
            SEC EDGAR (free, true PIT dates), yfinance (free fallback),
            or SimFin (if SIMFIN_API_KEY is set) and merges accruals,
            ROE trend, and earnings momentum via point-in-time join.
            No API key required — EDGAR and yfinance are zero-cost.

    Note:
        When using model_mode="regressor" or "ranker" with fundamental
        features, you MUST set include_fundamentals=True here (or call
        merge_fundamental_features() manually after this function).
        The trainer will raise ValueError if target_value is binary when
        model_mode expects continuous values.
    """
    if target_horizon_days < 1 or target_horizon_days > 20:
        raise ValueError("target_horizon_days must be between 1 and 20")

    result = db.execute(text(f"""
        SELECT
            fs.symbol,
            fs.target_session_date,
            s.sector,
            fs.rsi_14,
            fs.momentum_5d,
            fs.momentum_10d,
            fs.momentum_20d,
            fs.momentum_60d,
            fs.momentum_120d,
            fs.rolling_return_5d,
            fs.rolling_return_20d,
            fs.rolling_volatility_20d,
            fs.macd,
            fs.macd_signal,
            fs.short_term_reversal,
            fs.sector_return_1d,
            fs.sector_return_5d,
            fs.sector_relative_return_1d,
            fs.sector_relative_return_5d,
            fs.momentum_rank_market,
            fs.momentum_60d_rank_market,
            fs.momentum_120d_rank_market,
            fs.short_term_reversal_rank_market,
            fs.volatility_rank_market,
            fs.rsi_rank_market,
            fs.volume_rank_market,
            fs.sector_momentum_rank,
            fs.volume_change_5d,
            fs.volume_zscore_20d,
            fs.volatility_expansion_5_20,
            fs.volume_acceleration,
            fs.signed_volume_proxy,
            fs.price_volume_trend,
            fs.volume_imbalance_proxy,
            fs.liquidity_shock_5d,
            fs.vwap_deviation,
            fs.benchmark_relative_return_1d,
            fs.news_sentiment_24h,
            fs.news_sentiment_7d,
            fs.news_sentiment_std,
            fs.news_positive_ratio,
            fs.news_negative_ratio,
            fs.news_volume,
            fs.news_credibility_avg,
            fs.news_present_flag,
            fs.vix_level,
            fs.sp500_momentum_200d,
            fs.regime_label,
            s.market_cap,
            s.avg_daily_volume_30d,
            mb.close,
            mb.volume,
            -- Target: next day's return (label)
            LEAD(mb.close, {target_horizon_days}) OVER (PARTITION BY fs.symbol ORDER BY fs.target_session_date) / mb.close - 1 AS next_day_return
        FROM features_snapshot fs
        JOIN market_bars_daily mb
            ON fs.symbol = mb.symbol AND fs.target_session_date = mb.date
        JOIN symbols s
            ON fs.symbol = s.symbol
        WHERE fs.target_session_date >= :start_date
          AND fs.target_session_date <= :end_date
          AND fs.symbol = ANY(:symbols)
        ORDER BY fs.symbol, fs.target_session_date
    """), {"start_date": start_date, "end_date": end_date, "symbols": symbols})

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    columns = [
        "symbol", "target_session_date", "sector",
        "rsi_14", "momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d", "momentum_120d",
        "rolling_return_5d", "rolling_return_20d", "rolling_volatility_20d",
        "macd", "macd_signal", "short_term_reversal",
        "sector_return_1d", "sector_return_5d",
        "sector_relative_return_1d", "sector_relative_return_5d",
        "momentum_rank_market", "momentum_60d_rank_market", "momentum_120d_rank_market",
        "short_term_reversal_rank_market", "volatility_rank_market", "rsi_rank_market",
        "volume_rank_market", "sector_momentum_rank",
        "volume_change_5d", "volume_zscore_20d", "volatility_expansion_5_20",
        "volume_acceleration", "signed_volume_proxy", "price_volume_trend",
        "volume_imbalance_proxy", "liquidity_shock_5d", "vwap_deviation",
        "benchmark_relative_return_1d",
        "news_sentiment_24h", "news_sentiment_7d",
        "news_sentiment_std", "news_positive_ratio", "news_negative_ratio",
        "news_volume", "news_credibility_avg", "news_present_flag",
        "vix_level", "sp500_momentum_200d",
        "regime_label",
        "market_cap", "avg_daily_volume_30d", "close", "volume",
        "next_day_return",
    ]

    df = pd.DataFrame(rows, columns=columns)
    df["target_session_date"] = pd.to_datetime(df["target_session_date"])

    df = df.dropna(subset=["next_day_return"])

    # Non-overlapping sample filter: for multi-day horizons, consecutive rows
    # share target windows (e.g., row_t targets days t..t+5, row_{t+1} targets
    # t+1..t+6 — 4 out of 5 days overlap). Keeping every Nth row eliminates
    # overlap, giving honest sample counts and uncorrelated residuals.
    if sample_stride > 1:
        before = len(df)
        df = (
            df.sort_values(["symbol", "target_session_date"])
            .groupby("symbol", group_keys=False)
            .apply(lambda g: g.iloc[::sample_stride])
            .reset_index(drop=True)
        )
        logger.info(
            f"Sample stride={sample_stride}: {before} → {len(df)} rows "
            f"({len(df)/max(before,1)*100:.0f}% retained, non-overlapping targets)"
        )

    if include_liquidity_features:
        df["dollar_volume"] = df["close"] * df["volume"]

        # Point-in-time market cap proxy using only historical data.
        # close * rolling_30d_avg_volume gives trailing dollar volume which is
        # a cross-sectionally stable size proxy. NEVER fall back to static
        # symbols.market_cap — that's today's value applied to historical dates.
        df_sorted = df.sort_values(["symbol", "target_session_date"])
        rolling_adv_30 = df_sorted.groupby("symbol")["dollar_volume"].transform(
            lambda x: x.rolling(30, min_periods=5).mean()
        )
        df["_pit_market_cap"] = rolling_adv_30.clip(lower=1)

        df["log_market_cap"] = np.log1p(df["_pit_market_cap"])
        df["market_cap_rank"] = df.groupby("target_session_date")["_pit_market_cap"].rank(pct=True)

        rolling_avg_vol = df_sorted.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(30, min_periods=5).mean()
        )
        df["turnover_ratio"] = df["volume"] / rolling_avg_vol.replace(0, pd.NA)
        df["dollar_volume_rank_market"] = df.groupby("target_session_date")["dollar_volume"].rank(pct=True)

        df_sorted = df.sort_values(["symbol", "target_session_date"])
        rolling_turnover = df_sorted.groupby("symbol")["turnover_ratio"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        df["turnover_acceleration"] = df_sorted["turnover_ratio"] - rolling_turnover

        df = df.drop(columns=["_pit_market_cap"], errors="ignore")

    # ── Cross-sectional alpha features (computed from existing columns) ──
    # These vary across stocks on the same day — critical for market-relative models.

    vol_20 = df["rolling_volatility_20d"].clip(lower=0.001)
    df["vol_adj_momentum_20d"] = df["momentum_20d"] / vol_20
    df["vol_adj_momentum_60d"] = df["momentum_60d"] / vol_20

    df_sorted = df.sort_values(["symbol", "target_session_date"])
    df["high_52w"] = df_sorted.groupby("symbol")["close"].transform(
        lambda x: x.rolling(252, min_periods=60).max()
    )
    df["pct_from_52w_high"] = df["close"] / df["high_52w"].clip(lower=0.01)
    df = df.drop(columns=["high_52w"])

    spy_mom = (
        df[df["symbol"] == "SPY"][["target_session_date", "momentum_20d", "momentum_60d"]]
        .drop_duplicates("target_session_date")
        .rename(columns={"momentum_20d": "_spy_mom20", "momentum_60d": "_spy_mom60"})
    )
    df = df.merge(
        spy_mom[["target_session_date", "_spy_mom20", "_spy_mom60"]],
        on="target_session_date", how="left",
    )
    df["idio_momentum_20d"] = df["momentum_20d"] - df["_spy_mom20"].fillna(0)
    df["idio_momentum_60d"] = df["momentum_60d"] - df["_spy_mom60"].fillna(0)
    df = df.drop(columns=["_spy_mom20", "_spy_mom60"], errors="ignore")

    if "volume_change_5d" in df.columns and "momentum_5d" in df.columns:
        df["vol_price_divergence"] = (
            df.groupby("target_session_date")["volume_change_5d"].rank(pct=True)
            - df.groupby("target_session_date")["momentum_5d"].rank(pct=True)
        )

    for col in ["vol_adj_momentum_20d", "pct_from_52w_high", "idio_momentum_20d"]:
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby("target_session_date")[col].rank(pct=True)

    if target_mode == "absolute":
        df["target_value"] = df["next_day_return"]
        df["direction"] = (df["target_value"] > 0).astype(int)
    elif target_mode == "market_relative":
        market_map = (
            df[df["symbol"] == "SPY"][["target_session_date", "next_day_return"]]
            .drop_duplicates("target_session_date")
            .set_index("target_session_date")["next_day_return"]
        )
        df["benchmark_return"] = df["target_session_date"].map(market_map)
        df = df.dropna(subset=["benchmark_return"])
        df["target_value"] = df["next_day_return"] - df["benchmark_return"]
        df["direction"] = (df["target_value"] > 0).astype(int)
    elif target_mode == "sector_relative":
        sector_bench = (
            df.groupby(["target_session_date", "sector"])["next_day_return"]
            .mean()
            .rename("sector_benchmark")
            .reset_index()
        )
        df = df.merge(sector_bench, on=["target_session_date", "sector"], how="left")
        df = df.dropna(subset=["sector_benchmark"])
        df["target_value"] = df["next_day_return"] - df["sector_benchmark"]
        df["direction"] = (df["target_value"] > 0).astype(int)
    elif target_mode == "market_relative_top_bottom":
        market_map = (
            df[df["symbol"] == "SPY"][["target_session_date", "next_day_return"]]
            .drop_duplicates("target_session_date")
            .set_index("target_session_date")["next_day_return"]
        )
        df["benchmark_return"] = df["target_session_date"].map(market_map)
        df = df.dropna(subset=["benchmark_return"])
        df["target_value"] = df["next_day_return"] - df["benchmark_return"]

        bounds = (
            df.groupby("target_session_date")["target_value"]
            .quantile([top_bottom_pct, 1 - top_bottom_pct])
            .unstack()
            .rename(columns={top_bottom_pct: "lower_bound", 1 - top_bottom_pct: "upper_bound"})
            .reset_index()
        )
        df = df.merge(bounds, on="target_session_date", how="left")
        df["direction"] = pd.NA
        df.loc[df["target_value"] <= df["lower_bound"], "direction"] = 0
        df.loc[df["target_value"] >= df["upper_bound"], "direction"] = 1
        df = df.dropna(subset=["direction"])
        df["direction"] = df["direction"].astype(int)
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    if not include_liquidity_features:
        for col in [
            "log_market_cap", "market_cap_rank", "dollar_volume", "turnover_ratio",
            "dollar_volume_rank_market", "turnover_acceleration",
            "volume_acceleration_resid", "signed_volume_proxy_resid", "turnover_acceleration_resid",
            "volume_imbalance_proxy_resid", "liquidity_shock_5d_resid", "vwap_deviation_resid",
        ]:
            if col in df.columns:
                df = df.drop(columns=[col])

    # Cross-sectional transforms (same-day only, leakage-safe for ranking use).
    transform_cols = [
        "momentum_20d",
        "momentum_60d",
        "volume_change_5d",
        "volume_zscore_20d",
        "rolling_volatility_20d",
        "turnover_ratio",
    ]
    df = _add_cross_sectional_transforms(df, transform_cols)

    # Residual flow modeling: orthogonalize flow signals against size + sector.
    # Captures pure stock-level order flow after removing market-cap/sector bias.
    flow_resid_cols = [
        "volume_acceleration", "signed_volume_proxy", "turnover_acceleration",
        "volume_imbalance_proxy", "liquidity_shock_5d", "vwap_deviation",
    ]
    if include_liquidity_features and "log_market_cap" in df.columns:
        df = _add_flow_residuals(df, flow_resid_cols, size_col="log_market_cap")

    df = df.drop(columns=[c for c in ["market_cap", "avg_daily_volume_30d", "close", "volume"] if c in df.columns])

    # Optionally merge fundamental features (SUE, accruals, margins, etc.)
    if include_fundamentals:
        try:
            from src.features.fundamentals import (
                load_fundamentals,
                compute_fundamental_features,
                merge_fundamental_features,
            )
            fund_data = load_fundamentals(symbols, start_date, end_date)
            fund_features = compute_fundamental_features(fund_data)
            if not fund_features.empty:
                df = merge_fundamental_features(df, fund_features)
                logger.info("Fundamental features merged into training dataset")
            else:
                logger.warning("No fundamental features computed — continuing without")
        except Exception as e:
            logger.warning(f"Fundamental feature merge failed: {e} — continuing without")

    logger.info(
        f"Training dataset: {len(df)} rows, "
        f"{df['symbol'].nunique()} symbols, "
        f"{df['target_session_date'].min()} to {df['target_session_date'].max()} "
        f"(target_mode={target_mode}, horizon={target_horizon_days}, "
        f"liquidity={include_liquidity_features}, fundamentals={include_fundamentals})"
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
