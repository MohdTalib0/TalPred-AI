"""Expand trading universe from ~197 to S&P 500 stocks.

Pipeline:
  1. Fetch current S&P 500 list from Wikipedia (sector, company name, symbol)
  2. Fetch market cap + avg volume from yfinance in batches
  3. Upsert new symbols into the `symbols` table (skip existing)
  4. Report how many were added / already existed

Usage:
  python -m scripts.update_universe_sp500
  python -m scripts.update_universe_sp500 --dry-run   # preview without DB writes
"""

import argparse
import logging
import time
from datetime import date

import pandas as pd
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

# Map Wikipedia GICS sector names → our internal sector names
SECTOR_MAP = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
    "Consumer Discretionary": "Consumer Cyclical",
    "Financials": "Financial Services",
    "Industrials": "Industrials",
    "Consumer Staples": "Consumer Defensive",
    "Communication Services": "Communication Services",
    "Energy": "Energy",
    "Materials": "Basic Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}


def fetch_sp500_list() -> pd.DataFrame:
    """Download current S&P 500 constituent list from Wikipedia."""
    import io
    import requests

    logger.info("Fetching S&P 500 list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    sp500 = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
    sp500.columns = ["symbol", "company_name", "gics_sector"]

    # Wikipedia uses '.' where Yahoo Finance uses '-' (e.g. BRK.B → BRK-B)
    sp500["symbol"] = sp500["symbol"].str.replace(".", "-", regex=False)

    sp500["sector"] = sp500["gics_sector"].map(SECTOR_MAP)
    sp500 = sp500.dropna(subset=["symbol"])

    logger.info(f"  Found {len(sp500)} S&P 500 constituents")
    return sp500


def fetch_market_metrics(symbols: list[str], batch_size: int = 50) -> dict[str, dict]:
    """Get market_cap and avg_daily_volume from yfinance in batches."""
    logger.info(f"Fetching market metrics for {len(symbols)} symbols in batches of {batch_size}...")
    metrics: dict[str, dict] = {}

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        logger.info(f"  Batch {batch_num}/{total_batches}: {batch[0]} … {batch[-1]}")

        # Download 30 days to compute avg volume
        try:
            raw = yf.download(
                batch,
                period="30d",
                auto_adjust=False,
                threads=True,
                group_by="ticker",
                progress=False,
            )
        except Exception as e:
            logger.warning(f"  Download failed for batch {batch_num}: {e}")
            continue

        for sym in batch:
            try:
                sdf = raw[sym] if len(batch) > 1 else raw
                if sdf is None or sdf.empty:
                    continue
                avg_vol = float(sdf["Volume"].dropna().mean()) if "Volume" in sdf.columns else None
                last_close = float(sdf["Close"].dropna().iloc[-1]) if "Close" in sdf.columns and len(sdf) > 0 else None

                # Market cap from yfinance info (fallback: estimate from close × volume proxy)
                market_cap = None
                try:
                    info = yf.Ticker(sym).fast_info
                    market_cap = getattr(info, "market_cap", None)
                    if market_cap:
                        market_cap = float(market_cap)
                except Exception:
                    pass

                metrics[sym] = {
                    "market_cap": market_cap,
                    "avg_daily_volume_30d": avg_vol,
                    "last_close": last_close,
                }
            except Exception as e:
                logger.debug(f"  Skipped metrics for {sym}: {e}")

        time.sleep(0.5)  # be polite to yfinance

    logger.info(f"  Got metrics for {len(metrics)}/{len(symbols)} symbols")
    return metrics


def upsert_symbols(
    sp500_df: pd.DataFrame,
    metrics: dict[str, dict],
    dry_run: bool = False,
) -> dict:
    db = SessionLocal()
    existing = {row.symbol for row in db.query(Symbol.symbol).all()}

    added = 0
    updated = 0
    skipped = 0
    today = date.today()

    for _, row in sp500_df.iterrows():
        sym = row["symbol"]
        sector = row.get("sector")
        company = row.get("company_name", sym)
        m = metrics.get(sym, {})

        record = Symbol(
            symbol=sym,
            company_name=str(company)[:255],
            exchange="NYSE",  # S&P 500 is NYSE/NASDAQ; we simplify
            sector=sector,
            market_cap=m.get("market_cap"),
            avg_daily_volume_30d=m.get("avg_daily_volume_30d"),
            is_active=True,
            effective_from=today,
            effective_to=None,
        )

        if sym in existing:
            # Update sector and metrics for existing symbols
            existing_row = db.query(Symbol).filter(Symbol.symbol == sym).first()
            if existing_row:
                if sector and not existing_row.sector:
                    existing_row.sector = sector
                if m.get("market_cap"):
                    existing_row.market_cap = m["market_cap"]
                if m.get("avg_daily_volume_30d"):
                    existing_row.avg_daily_volume_30d = m["avg_daily_volume_30d"]
                existing_row.is_active = True
                updated += 1
        else:
            if not dry_run:
                db.add(record)
            added += 1

        if sym not in existing and sym not in [s.symbol for s in db.new]:
            # Already counted
            pass

    if not dry_run:
        db.commit()
    db.close()

    return {"added": added, "updated": updated, "total_sp500": len(sp500_df), "existing_before": len(existing)}


def main():
    parser = argparse.ArgumentParser(description="Expand universe to S&P 500")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip yfinance metric fetch (faster)")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN mode — no DB writes")

    # Step 1: Get S&P 500 list
    sp500_df = fetch_sp500_list()
    symbols = sp500_df["symbol"].tolist()
    logger.info(f"\nSectors distribution:\n{sp500_df['sector'].value_counts().to_string()}")

    # Step 2: Fetch market metrics
    if args.skip_metrics:
        logger.info("Skipping market metrics (--skip-metrics)")
        metrics = {}
    else:
        metrics = fetch_market_metrics(symbols, batch_size=50)

    # Step 3: Upsert
    logger.info("\nUpserting symbols to database...")
    result = upsert_symbols(sp500_df, metrics, dry_run=args.dry_run)

    logger.info("\n" + "=" * 50)
    logger.info("UNIVERSE EXPANSION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  S&P 500 constituents:  {result['total_sp500']}")
    logger.info(f"  Already in DB:         {result['existing_before']}")
    logger.info(f"  Newly added:           {result['added']}")
    logger.info(f"  Updated (metrics):     {result['updated']}")
    if args.dry_run:
        logger.info("  (DRY RUN — nothing written)")
    else:
        logger.info("\nNext steps:")
        logger.info("  1. python -m scripts.backfill_market_fast --years 10")
        logger.info("  2. python -m scripts.backfill_features --start 2015-01-01 --end today")


if __name__ == "__main__":
    main()
