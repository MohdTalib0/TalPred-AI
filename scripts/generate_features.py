"""Feature generation script.

Modes:
  - incremental (default): compute features for latest trading session only
  - backfill: compute features for all historical dates

Usage:
  python -m scripts.generate_features                    # incremental
  python -m scripts.generate_features --mode backfill    # full history
"""

import argparse
import logging
import time

from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.features.engine import (  # noqa: E402
    compute_sector_returns,
    generate_features,
    load_market_window,
    load_sector_map,
    save_sector_returns,
    save_snapshots,
)
from src.models.schema import Symbol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)


def get_active_symbols(db) -> list[str]:
    return [row.symbol for row in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]


def get_all_trading_dates(db) -> list:
    result = db.execute(text("""
        SELECT DISTINCT date FROM market_bars_daily ORDER BY date
    """))
    return [row[0] for row in result.fetchall()]


def main():
    parser = argparse.ArgumentParser(description="Feature generation")
    parser.add_argument("--mode", choices=["incremental", "backfill"], default="incremental")
    parser.add_argument("--dataset-version", type=str, default=None)
    args = parser.parse_args()

    db = SessionLocal()
    t0 = time.time()

    try:
        symbols = get_active_symbols(db)
        logger.info(f"Active symbols: {len(symbols)}")

        if args.mode == "backfill":
            trading_dates = get_all_trading_dates(db)
            logger.info(f"Backfill mode: {len(trading_dates)} trading dates")

            # Sector returns
            logger.info("Computing sector returns...")
            market_df = load_market_window(db, symbols, lookback_days=2000)
            sector_map = load_sector_map(db)
            sector_df = compute_sector_returns(market_df, sector_map)
            save_sector_returns(db, sector_df)

            # Features in chunks to manage memory
            chunk_size = 50
            total_saved = 0
            for i in range(0, len(trading_dates), chunk_size):
                chunk_dates = trading_dates[i : i + chunk_size]
                logger.info(f"  Processing dates {i+1}-{i+len(chunk_dates)} of {len(trading_dates)}")
                snapshots = generate_features(
                    db, symbols,
                    target_dates=chunk_dates,
                    dataset_version=args.dataset_version,
                )
                total_saved += save_snapshots(db, snapshots)

            logger.info(f"Backfill complete: {total_saved} snapshots in {time.time() - t0:.1f}s")

        else:
            # Incremental: latest date only
            logger.info("Incremental mode: latest session only")
            snapshots = generate_features(
                db, symbols,
                target_dates=None,
                dataset_version=args.dataset_version,
            )
            saved = save_snapshots(db, snapshots)
            logger.info(f"Incremental complete: {saved} snapshots in {time.time() - t0:.1f}s")

    finally:
        db.close()


if __name__ == "__main__":
    main()
