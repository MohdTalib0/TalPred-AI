"""Production-grade market data backfill using COPY + staging table merge.

Pipeline:
  1. yf.download() bulk fetch (all symbols, parallel threads)
  2. pandas → CSV buffer
  3. PostgreSQL COPY into unindexed staging table
  4. SQL merge from staging into production table
  5. Truncate staging

Expected: 245k rows in 10-30 seconds.
Usage: python -m scripts.backfill_market_fast [--years 5]
"""

import argparse
import logging
import os
import sys
import time
from datetime import UTC, date, datetime, timedelta
from io import StringIO

import pandas as pd
import psycopg2
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.models.schema import Symbol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

COLUMNS = [
    "symbol", "date", "open", "high", "low", "close",
    "adj_close", "volume", "source", "event_time", "as_of_time",
]

MERGE_SQL = """
    INSERT INTO market_bars_daily
        (symbol, date, open, high, low, close, adj_close, volume, source, event_time, as_of_time)
    SELECT symbol, date, open, high, low, close, adj_close, volume, source, event_time, as_of_time
    FROM market_bars_staging
    ON CONFLICT (symbol, date) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        adj_close = EXCLUDED.adj_close,
        volume = EXCLUDED.volume,
        as_of_time = EXCLUDED.as_of_time;
"""


def get_symbols() -> list[str]:
    db = SessionLocal()
    symbols = [
        row.symbol
        for row in db.query(Symbol).filter(Symbol.is_active.is_(True)).order_by(Symbol.symbol).all()
    ]
    db.close()
    return symbols


def download_all(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    logger.info(f"Step 1: Downloading {len(symbols)} symbols from yfinance...")
    t0 = time.time()

    df = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=False,
        threads=True,
        group_by="ticker",
    )

    logger.info(f"  Download complete in {time.time() - t0:.1f}s, shape: {df.shape}")
    return df


def reshape_to_dataframe(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    logger.info("Step 2: Reshaping to flat DataFrame...")
    t0 = time.time()
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S+00")
    rows = []

    for symbol in symbols:
        try:
            sdf = df[symbol].copy() if len(symbols) > 1 else df.copy()
            sdf = sdf.dropna(subset=["Close"])
            if sdf.empty:
                continue

            for idx, row in sdf.iterrows():
                bar_date = pd.Timestamp(idx).date()
                high = float(row["High"]) if pd.notna(row["High"]) else None
                low = float(row["Low"]) if pd.notna(row["Low"]) else None
                vol = float(row["Volume"]) if pd.notna(row["Volume"]) else 0

                if high is not None and low is not None and high < low:
                    continue
                if vol < 0:
                    continue

                rows.append([
                    symbol,
                    bar_date.isoformat(),
                    float(row["Open"]) if pd.notna(row["Open"]) else "",
                    high if high is not None else "",
                    low if low is not None else "",
                    float(row["Close"]) if pd.notna(row["Close"]) else "",
                    float(row["Adj Close"]) if pd.notna(row.get("Adj Close")) else "",
                    vol,
                    "yfinance",
                    f"{bar_date.isoformat()} 00:00:00+00",
                    now,
                ])
        except Exception:
            logger.warning(f"  Skipped {symbol}")

    logger.info(f"  Reshaped {len(rows):,} records in {time.time() - t0:.1f}s")
    return rows


def build_csv_buffer(rows: list[list]) -> StringIO:
    logger.info("Step 3: Building CSV buffer...")
    t0 = time.time()
    buf = StringIO()
    for row in rows:
        buf.write("\t".join(str(v) for v in row) + "\n")
    buf.seek(0)
    size_mb = buf.tell() / (1024 * 1024)
    buf.seek(0)
    logger.info(f"  CSV buffer: {size_mb:.1f} MB in {time.time() - t0:.1f}s")
    return buf


def copy_and_merge(buf: StringIO, row_count: int):
    db_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    try:
        # Truncate staging
        logger.info("Step 4: COPY into staging table...")
        t0 = time.time()
        cur.execute("TRUNCATE market_bars_staging")

        # COPY CSV into staging (tab-delimited, empty string = NULL)
        cur.copy_expert(
            """
            COPY market_bars_staging
                (symbol, date, open, high, low, close, adj_close, volume, source, event_time, as_of_time)
            FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')
            """,
            buf,
        )
        conn.commit()
        logger.info(f"  COPY complete: {row_count:,} rows in {time.time() - t0:.1f}s")

        # Verify staging
        cur.execute("SELECT count(*) FROM market_bars_staging")
        staging_count = cur.fetchone()[0]
        logger.info(f"  Staging table: {staging_count:,} rows")

        # Merge staging → production
        logger.info("Step 5: Merging staging → market_bars_daily...")
        t1 = time.time()
        cur.execute(MERGE_SQL)
        conn.commit()
        logger.info(f"  Merge complete in {time.time() - t1:.1f}s")

        # Verify production
        cur.execute("SELECT count(*) FROM market_bars_daily")
        prod_count = cur.fetchone()[0]
        logger.info(f"  Production table: {prod_count:,} rows")

        # Clean up staging
        cur.execute("TRUNCATE market_bars_staging")
        conn.commit()

    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Fast market data backfill (COPY pipeline)")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--truncate", action="store_true", help="Truncate production table first")
    args = parser.parse_args()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.years * 365)

    symbols = get_symbols()
    if not symbols:
        logger.error("No active symbols. Run seed_symbols first.")
        sys.exit(1)

    logger.info(f"Backfill: {len(symbols)} symbols, {start_date} → {end_date}")
    t_total = time.time()

    if args.truncate:
        db_url = os.environ["DATABASE_URL"]
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("TRUNCATE market_bars_daily")
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Truncated market_bars_daily")

    df = download_all(symbols, start_date, end_date)
    rows = reshape_to_dataframe(df, symbols)
    buf = build_csv_buffer(rows)
    copy_and_merge(buf, len(rows))

    elapsed = time.time() - t_total
    logger.info(f"Total backfill: {len(rows):,} rows in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
