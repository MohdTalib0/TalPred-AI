"""Backfill features for training window.

Generates feature snapshots in batches for the ML training pipeline.
Uses bulk insert via psycopg2 COPY for performance.

Usage:
  python -m scripts.backfill_features --start 2023-06-01 --end 2026-03-07
"""

import argparse
import io
import logging
import time
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.config import settings  # noqa: E402
from src.db import SessionLocal  # noqa: E402
from src.features.engine import (  # noqa: E402
    generate_features,
    save_snapshots,
)
from src.models.schema import Symbol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("backfill_features")


def get_trading_dates(db, start: date, end: date) -> list[date]:
    """Get actual trading dates from market_bars_daily."""
    from sqlalchemy import text
    result = db.execute(text("""
        SELECT DISTINCT date FROM market_bars_daily
        WHERE date >= :start AND date <= :end
        ORDER BY date
    """), {"start": start, "end": end})
    return [row[0] for row in result.fetchall()]


def bulk_save_snapshots(db, snapshots: list[dict]) -> int:
    """Save snapshots using psycopg2 COPY for speed."""
    if not snapshots:
        return 0

    import psycopg2

    columns = [
        "snapshot_id", "symbol", "as_of_time", "target_session_date",
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
        "regime_label", "dataset_version",
    ]

    buf = io.StringIO()
    for snap in snapshots:
        vals = []
        for col in columns:
            v = snap.get(col)
            if v is None:
                vals.append("\\N")
            else:
                vals.append(str(v).replace("\t", " ").replace("\n", " "))
        buf.write("\t".join(vals) + "\n")
    buf.seek(0)

    dsn = settings.database_url.replace("+psycopg2", "").replace("postgresql://", "postgresql://")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    col_list = ", ".join(columns)
    cur.execute(f"""
        CREATE TEMP TABLE features_staging (LIKE features_snapshot INCLUDING ALL)
        ON COMMIT DROP
    """)
    cur.copy_expert(
        f"COPY features_staging ({col_list}) FROM STDIN WITH (FORMAT text, NULL '\\N')",
        buf,
    )
    cur.execute(f"""
        INSERT INTO features_snapshot ({col_list})
        SELECT {col_list} FROM features_staging
        ON CONFLICT (snapshot_id) DO UPDATE SET
            rsi_14 = EXCLUDED.rsi_14,
            momentum_5d = EXCLUDED.momentum_5d,
            momentum_10d = EXCLUDED.momentum_10d,
            momentum_20d = EXCLUDED.momentum_20d,
            momentum_60d = EXCLUDED.momentum_60d,
            momentum_120d = EXCLUDED.momentum_120d,
            rolling_return_5d = EXCLUDED.rolling_return_5d,
            rolling_return_20d = EXCLUDED.rolling_return_20d,
            rolling_volatility_20d = EXCLUDED.rolling_volatility_20d,
            macd = EXCLUDED.macd,
            macd_signal = EXCLUDED.macd_signal,
            short_term_reversal = EXCLUDED.short_term_reversal,
            sector_return_1d = EXCLUDED.sector_return_1d,
            sector_return_5d = EXCLUDED.sector_return_5d,
            sector_relative_return_1d = EXCLUDED.sector_relative_return_1d,
            sector_relative_return_5d = EXCLUDED.sector_relative_return_5d,
            momentum_rank_market = EXCLUDED.momentum_rank_market,
            momentum_60d_rank_market = EXCLUDED.momentum_60d_rank_market,
            momentum_120d_rank_market = EXCLUDED.momentum_120d_rank_market,
            short_term_reversal_rank_market = EXCLUDED.short_term_reversal_rank_market,
            volatility_rank_market = EXCLUDED.volatility_rank_market,
            rsi_rank_market = EXCLUDED.rsi_rank_market,
            volume_rank_market = EXCLUDED.volume_rank_market,
            sector_momentum_rank = EXCLUDED.sector_momentum_rank,
            volume_change_5d = EXCLUDED.volume_change_5d,
            volume_zscore_20d = EXCLUDED.volume_zscore_20d,
            volatility_expansion_5_20 = EXCLUDED.volatility_expansion_5_20,
            volume_acceleration = EXCLUDED.volume_acceleration,
            signed_volume_proxy = EXCLUDED.signed_volume_proxy,
            price_volume_trend = EXCLUDED.price_volume_trend,
            volume_imbalance_proxy = EXCLUDED.volume_imbalance_proxy,
            liquidity_shock_5d = EXCLUDED.liquidity_shock_5d,
            vwap_deviation = EXCLUDED.vwap_deviation,
            benchmark_relative_return_1d = EXCLUDED.benchmark_relative_return_1d,
            news_sentiment_24h = EXCLUDED.news_sentiment_24h,
            news_sentiment_7d = EXCLUDED.news_sentiment_7d,
            news_sentiment_std = EXCLUDED.news_sentiment_std,
            news_positive_ratio = EXCLUDED.news_positive_ratio,
            news_negative_ratio = EXCLUDED.news_negative_ratio,
            news_volume = EXCLUDED.news_volume,
            news_credibility_avg = EXCLUDED.news_credibility_avg,
            news_present_flag = EXCLUDED.news_present_flag,
            vix_level = EXCLUDED.vix_level,
            sp500_momentum_200d = EXCLUDED.sp500_momentum_200d,
            regime_label = EXCLUDED.regime_label,
            dataset_version = EXCLUDED.dataset_version
    """)
    count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return count


def main():
    parser = argparse.ArgumentParser(description="Backfill feature snapshots")
    parser.add_argument("--start", type=str, default="2023-06-01")
    parser.add_argument("--end", type=str, default="2026-03-07")
    parser.add_argument("--batch-days", type=int, default=60,
                        help="Process this many trading days per batch")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    db = SessionLocal()
    t0 = time.time()

    symbols = [
        row.symbol for row in
        db.query(Symbol).filter(Symbol.is_active.is_(True)).all()
    ]
    logger.info(f"Backfilling features for {len(symbols)} symbols from {start} to {end}")

    trading_dates = get_trading_dates(db, start, end)
    logger.info(f"Found {len(trading_dates)} trading days")

    total_saved = 0
    for batch_start in range(0, len(trading_dates), args.batch_days):
        batch_dates = trading_dates[batch_start:batch_start + args.batch_days]
        batch_num = batch_start // args.batch_days + 1
        total_batches = (len(trading_dates) + args.batch_days - 1) // args.batch_days

        bt0 = time.time()
        logger.info(
            f"Batch {batch_num}/{total_batches}: "
            f"{batch_dates[0]} to {batch_dates[-1]} ({len(batch_dates)} days)"
        )

        lookback = (date.today() - batch_dates[0]).days + 310
        snapshots = generate_features(
            db, symbols,
            target_dates=batch_dates,
            lookback_days=lookback,
            dataset_version="v1.0-backfill",
        )
        if snapshots:
            saved = bulk_save_snapshots(db, snapshots)
            total_saved += saved
            logger.info(
                f"  Saved {saved} snapshots in {time.time() - bt0:.1f}s "
                f"(total: {total_saved})"
            )
        else:
            logger.warning("  No snapshots generated for this batch")

    elapsed = time.time() - t0
    logger.info(f"Feature backfill complete: {total_saved} snapshots in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    db.close()


if __name__ == "__main__":
    main()
