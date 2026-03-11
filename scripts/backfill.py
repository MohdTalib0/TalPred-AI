"""Historical data backfill script (DE-107).

Backfills market calendar, market bars, and macro series for 3-5 years.
Usage: python -m scripts.backfill [--years 5] [--step calendar|market|macro|all]
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from src.calendar.service import sync_calendar_to_db  # noqa: E402
from src.connectors.macro import fetch_macro_series  # noqa: E402
from src.db import SessionLocal  # noqa: E402
from src.models.schema import MacroSeries, Symbol  # noqa: E402
from src.pipelines.ingest_market import ingest_symbol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MACRO_SERIES = [
    "DFF",        # Federal Funds Rate
    "CPIAUCSL",   # CPI
    "DCOILWTICO", # WTI Crude Oil
    "DTWEXBGS",   # Trade Weighted USD
    "VIXCLS",     # VIX
]


def backfill_calendar(db, start: date, end: date):
    logger.info(f"Backfilling market calendar: {start} -> {end}")

    for exchange in ["NYSE", "NASDAQ"]:
        count = sync_calendar_to_db(db, start, end, exchange)
        logger.info(f"  {exchange}: {count} calendar entries upserted")


def backfill_market(db, start: date, end: date, batch_size: int = 10):
    symbols = [row.symbol for row in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]

    if not symbols:
        logger.error("No active symbols in DB. Run seed_symbols first.")
        return

    logger.info(f"Backfilling market bars for {len(symbols)} symbols: {start} -> {end}")

    total_inserted = 0
    total_failed = []

    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"[{i}/{len(symbols)}] {symbol}")
            result = ingest_symbol(db, symbol, start, end)
            total_inserted += result["inserted"] + result["updated"]

            if i % batch_size == 0:
                logger.info(f"  Progress: {i}/{len(symbols)}, total rows: {total_inserted}")
                time.sleep(1)

        except Exception:
            logger.exception(f"  Failed: {symbol}")
            total_failed.append(symbol)
            time.sleep(2)

    logger.info(
        f"Market backfill complete: {total_inserted} rows, "
        f"{len(total_failed)} failed: {total_failed}"
    )


def backfill_macro(db, start: date, end: date):
    logger.info(f"Backfilling macro series: {start} -> {end}")

    for series_id in DEFAULT_MACRO_SERIES:
        logger.info(f"  Fetching {series_id}...")
        records = fetch_macro_series(series_id, start, end)

        upserted = 0
        for rec in records:
            existing = (
                db.query(MacroSeries)
                .filter(
                    MacroSeries.series_id == rec["series_id"],
                    MacroSeries.observation_date == rec["observation_date"],
                )
                .first()
            )

            if existing:
                existing.value = rec["value"]
                existing.release_time_utc = rec["release_time_utc"]
                existing.available_at_utc = rec["available_at_utc"]
            else:
                db.add(MacroSeries(**rec))
            upserted += 1

        db.commit()
        logger.info(f"  {series_id}: {upserted} observations upserted")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Historical data backfill")
    parser.add_argument("--years", type=int, default=5, help="Years to backfill (default: 5)")
    parser.add_argument(
        "--step",
        choices=["calendar", "market", "macro", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.years * 365)

    logger.info(f"Backfill range: {start_date} -> {end_date} ({args.years} years)")

    db = SessionLocal()
    try:
        if args.step in ("calendar", "all"):
            backfill_calendar(db, start_date, end_date)

        if args.step in ("market", "all"):
            backfill_market(db, start_date, end_date)

        if args.step in ("macro", "all"):
            backfill_macro(db, start_date, end_date)

        logger.info("Backfill complete.")

    except Exception:
        logger.exception("Backfill failed")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
