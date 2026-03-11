"""Seed the symbols table with the US Large Cap 200 universe.

Fetches metadata from yfinance for each ticker and upserts into the symbols table.
Usage: python -m scripts.seed_symbols
"""

import logging
import sys
from datetime import date

import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.models.schema import Symbol  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

US_LARGE_CAP_200 = [
    # Benchmark ETF (S&P 500 proxy)
    "SPY",
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CSCO",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "INTU", "AMAT", "NOW", "IBM",
    "MU", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "FTNT", "PANW", "CRWD",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
    "C", "AXP", "SCHW", "CB", "MMC", "ICE", "CME", "PGR", "AON", "MET",
    "AFL", "AIG", "TRV", "ALL", "PRU",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "BMY", "GILD", "MDT", "ELV", "ISRG", "VRTX", "SYK", "BSX", "REGN", "ZTS",
    "CI", "HCA", "BDX", "MCK", "EW",
    # Consumer Discretionary
    "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR", "CMG", "ORLY",
    "AZO", "ROST", "DHI", "LEN", "GM", "F", "YUM", "DARDEN", "HLT", "EBAY",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "GIS",
    "KMB", "SYY", "KHC", "HSY", "STZ", "KDP", "MKC", "CHD", "K", "TSN",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB",
    "KMI", "HAL", "DVN", "HES", "FANG",
    # Industrials
    "CAT", "RTX", "UNP", "HON", "DE", "GE", "BA", "LMT", "UPS", "MMM",
    "ETN", "ITW", "EMR", "NOC", "GD", "WM", "RSG", "CSX", "NSC", "CTAS",
    "PCAR", "TT", "CARR", "AME", "ROK",
    # Communication Services
    "GOOG", "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ED",
    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    # Materials
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "DD", "DOW", "VMC",
]


def fetch_ticker_info(ticker: str) -> dict | None:
    """Fetch company metadata from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("regularMarketPrice") is None:
            return None
        return {
            "symbol": ticker,
            "company_name": info.get("shortName", info.get("longName", ticker)),
            "exchange": info.get("exchange", "UNKNOWN"),
            "sector": info.get("sector"),
            "market_cap": info.get("marketCap"),
            "avg_daily_volume_30d": info.get("averageDailyVolume10Day"),
        }
    except Exception as e:
        logger.warning(f"Failed to fetch {ticker}: {e}")
        return None


def seed_symbols():
    db = SessionLocal()
    try:
        existing = {row[0] for row in db.execute(text("SELECT symbol FROM symbols")).fetchall()}
        logger.info(f"{len(existing)} symbols already in DB")

        added, skipped, failed = 0, 0, 0
        total = len(US_LARGE_CAP_200)

        for i, ticker in enumerate(US_LARGE_CAP_200, 1):
            if ticker in existing:
                skipped += 1
                continue

            logger.info(f"[{i}/{total}] Fetching {ticker}...")
            info = fetch_ticker_info(ticker)

            if info is None:
                logger.warning(f"  Skipping {ticker} - no data")
                failed += 1
                continue

            db.add(Symbol(
                symbol=info["symbol"],
                company_name=info["company_name"],
                exchange=info["exchange"],
                sector=info["sector"],
                market_cap=info["market_cap"],
                avg_daily_volume_30d=info["avg_daily_volume_30d"],
                is_active=True,
                effective_from=date(2021, 1, 1),
            ))
            added += 1

            if added % 20 == 0:
                db.commit()
                logger.info(f"  Committed batch ({added} added so far)")

        db.commit()
        logger.info(f"Seeding complete: {added} added, {skipped} existed, {failed} failed")

    except Exception as e:
        db.rollback()
        logger.error(f"Seeding failed: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    seed_symbols()
