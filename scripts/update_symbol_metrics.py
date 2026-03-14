"""Backfill null market_cap and avg_daily_volume_30d for symbols in the DB.

Uses yfinance:
  - batch download (30 days) → avg_daily_volume_30d
  - fast_info per ticker      → market_cap

Only updates rows where at least one field is NULL (safe to re-run).

Usage:
  python -m scripts.update_symbol_metrics           # only null-metric symbols
  python -m scripts.update_symbol_metrics --all     # refresh all symbols
  python -m scripts.update_symbol_metrics --dry-run # preview, no writes
"""

import argparse
import logging
import os
import time
from datetime import date, timedelta

import pandas as pd
import psycopg2
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal
from src.models.schema import Symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

BATCH = 50  # symbols per yfinance download call


def get_symbols_to_fix(refresh_all: bool) -> list[str]:
    db = SessionLocal()
    q = db.query(Symbol.symbol).filter(Symbol.is_active.is_(True))
    if not refresh_all:
        q = q.filter(
            (Symbol.market_cap.is_(None)) | (Symbol.avg_daily_volume_30d.is_(None))
        )
    symbols = [row.symbol for row in q.order_by(Symbol.symbol).all()]
    db.close()
    logger.info(f"Symbols to update: {len(symbols)}")
    return symbols


def fetch_volume_from_yf(symbols: list[str]) -> dict[str, float]:
    """Download 30 days OHLCV and compute mean daily volume per symbol."""
    end = date.today()
    start = end - timedelta(days=45)  # grab 45 days to ensure 30 trading days

    try:
        raw = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=False,
            threads=True,
            group_by="ticker",
            progress=False,
        )
    except Exception as e:
        logger.warning(f"  yf.download error: {e}")
        return {}

    vols = {}
    for sym in symbols:
        try:
            sdf = raw[sym] if len(symbols) > 1 else raw
            if sdf is None or sdf.empty or "Volume" not in sdf.columns:
                continue
            v = sdf["Volume"].dropna()
            if len(v) > 0:
                vols[sym] = float(v.mean())
        except Exception:
            pass
    return vols


def fetch_market_caps_from_yf(symbols: list[str]) -> dict[str, float]:
    """Use yfinance fast_info for market cap (much faster than .info)."""
    caps = {}
    for sym in symbols:
        try:
            fi = yf.Ticker(sym).fast_info
            mc = getattr(fi, "market_cap", None)
            if mc and float(mc) > 0:
                caps[sym] = float(mc)
        except Exception:
            pass
    return caps


def update_db(updates: dict[str, dict], dry_run: bool):
    """Bulk-update symbols table using psycopg2 for speed."""
    if not updates:
        logger.info("Nothing to update.")
        return

    db_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    updated = 0
    for sym, vals in updates.items():
        if dry_run:
            logger.debug(f"  [DRY] {sym}: {vals}")
            updated += 1
            continue

        sets = []
        params = []
        if vals.get("market_cap") is not None:
            sets.append("market_cap = %s")
            params.append(vals["market_cap"])
        if vals.get("avg_daily_volume_30d") is not None:
            sets.append("avg_daily_volume_30d = %s")
            params.append(vals["avg_daily_volume_30d"])
        if not sets:
            continue

        params.append(sym)
        cur.execute(
            f"UPDATE symbols SET {', '.join(sets)} WHERE symbol = %s",
            params,
        )
        updated += 1

    if not dry_run:
        conn.commit()
    cur.close()
    conn.close()
    logger.info(f"  {'[DRY] Would update' if dry_run else 'Updated'} {updated} symbols")


def main():
    parser = argparse.ArgumentParser(description="Fill null market_cap / avg_daily_volume for symbols")
    parser.add_argument("--all", action="store_true", help="Refresh all symbols, not just null-metric ones")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--batch", type=int, default=BATCH, help="Symbols per yfinance batch")
    args = parser.parse_args()

    symbols = get_symbols_to_fix(refresh_all=args.all)
    if not symbols:
        logger.info("No symbols need updating. Done.")
        return

    total_batches = (len(symbols) + args.batch - 1) // args.batch
    all_updates: dict[str, dict] = {}

    for i in range(0, len(symbols), args.batch):
        batch = symbols[i : i + args.batch]
        b_num = i // args.batch + 1
        logger.info(f"Batch {b_num}/{total_batches}: {batch[0]} … {batch[-1]} ({len(batch)} symbols)")

        # 1 — avg daily volume from 30-day price download
        vols = fetch_volume_from_yf(batch)
        logger.info(f"  Volume: got {len(vols)}/{len(batch)}")

        # 2 — market cap from fast_info
        caps = fetch_market_caps_from_yf(batch)
        logger.info(f"  Market cap: got {len(caps)}/{len(batch)}")

        for sym in batch:
            entry: dict = {}
            if sym in vols:
                entry["avg_daily_volume_30d"] = vols[sym]
            if sym in caps:
                entry["market_cap"] = caps[sym]
            if entry:
                all_updates[sym] = entry

        time.sleep(1.0)  # be polite to yfinance

    logger.info(f"\nTotal symbols with data to write: {len(all_updates)}")
    logger.info("Writing to database...")
    update_db(all_updates, dry_run=args.dry_run)

    if not args.dry_run:
        # Quick sanity check
        db = SessionLocal()
        null_cap = db.query(Symbol).filter(
            Symbol.is_active.is_(True), Symbol.market_cap.is_(None)
        ).count()
        null_vol = db.query(Symbol).filter(
            Symbol.is_active.is_(True), Symbol.avg_daily_volume_30d.is_(None)
        ).count()
        db.close()
        logger.info(f"\nRemaining NULLs after update:")
        logger.info(f"  market_cap NULL:          {null_cap}")
        logger.info(f"  avg_daily_volume_30d NULL: {null_vol}")
        if null_cap == 0 and null_vol == 0:
            logger.info("  All clean!")
        else:
            logger.info("  (Some may be unavailable from yfinance — typically ETFs / foreign tickers)")


if __name__ == "__main__":
    main()
