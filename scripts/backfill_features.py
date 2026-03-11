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
        "rsi_14", "momentum_5d", "momentum_10d",
        "rolling_return_5d", "rolling_return_20d", "rolling_volatility_20d",
        "macd", "macd_signal",
        "sector_return_1d", "sector_return_5d",
        "benchmark_relative_return_1d",
        "news_sentiment_24h", "news_sentiment_7d",
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
            rolling_return_5d = EXCLUDED.rolling_return_5d,
            rolling_return_20d = EXCLUDED.rolling_return_20d,
            rolling_volatility_20d = EXCLUDED.rolling_volatility_20d,
            macd = EXCLUDED.macd,
            macd_signal = EXCLUDED.macd_signal,
            sector_return_1d = EXCLUDED.sector_return_1d,
            sector_return_5d = EXCLUDED.sector_return_5d,
            benchmark_relative_return_1d = EXCLUDED.benchmark_relative_return_1d,
            news_sentiment_24h = EXCLUDED.news_sentiment_24h,
            news_sentiment_7d = EXCLUDED.news_sentiment_7d,
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
        snapshots = generate_features(db, symbols, target_dates=batch_dates, lookback_days=lookback)
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
