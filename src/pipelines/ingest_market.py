"""Market data ingestion pipeline.

Fetches daily bars via yfinance, validates them, and upserts into market_bars_daily.
Invalid records are routed to the quarantine table.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from src.connectors.market import fetch_daily_bars
from src.db import SessionLocal
from src.models.schema import MarketBarsDaily, QuarantineRecord

logger = logging.getLogger(__name__)


def validate_bar(row: dict) -> tuple[bool, str]:
    """Validate a single market bar row.

    Returns (is_valid, reason).
    """
    if row.get("high") is not None and row.get("low") is not None:
        if row["high"] < row["low"]:
            return False, "high < low"

    if row.get("volume") is not None and row["volume"] < 0:
        return False, "negative volume"

    required = ["symbol", "date", "open", "high", "low", "close"]
    for field in required:
        if row.get(field) is None:
            return False, f"missing required field: {field}"

    return True, ""


def ingest_symbol(
    db: Session,
    symbol: str,
    start_date: date,
    end_date: date,
) -> dict:
    """Ingest daily bars for a single symbol.

    Returns dict with counts: inserted, updated, quarantined.
    """
    df = fetch_daily_bars(symbol, start_date, end_date)

    if df.empty:
        logger.warning(f"No data for {symbol}")
        return {"inserted": 0, "updated": 0, "quarantined": 0}

    inserted = 0
    updated = 0
    quarantined = 0

    for _, row in df.iterrows():
        record = row.to_dict()
        is_valid, reason = validate_bar(record)

        if not is_valid:
            db.add(QuarantineRecord(
                source="market_bars_daily",
                record_data=record,
                failure_reason=reason,
            ))
            quarantined += 1
            continue

        existing = (
            db.query(MarketBarsDaily)
            .filter(
                MarketBarsDaily.symbol == record["symbol"],
                MarketBarsDaily.date == record["date"],
            )
            .first()
        )

        if existing:
            existing.open = record["open"]
            existing.high = record["high"]
            existing.low = record["low"]
            existing.close = record["close"]
            existing.adj_close = record.get("adj_close")
            existing.volume = record["volume"]
            existing.as_of_time = datetime.now(UTC)
            updated += 1
        else:
            db.add(MarketBarsDaily(**record))
            inserted += 1

    db.commit()
    logger.debug(
        f"{symbol}: inserted={inserted}, updated={updated}, quarantined={quarantined}"
    )
    return {"inserted": inserted, "updated": updated, "quarantined": quarantined}


def _ingest_symbol_isolated(
    symbol: str,
    start_date: date,
    end_date: date,
) -> tuple[str, dict]:
    """Ingest one symbol with an isolated DB session (thread-safe)."""
    db = SessionLocal()
    try:
        return symbol, ingest_symbol(db, symbol, start_date, end_date)
    finally:
        db.close()


def ingest_universe(
    db: Session,
    symbols: list[str],
    start_date: date,
    end_date: date,
    max_workers: int = 1,
) -> dict:
    """Ingest daily bars for entire symbol universe."""
    totals = {"inserted": 0, "updated": 0, "quarantined": 0, "failed_symbols": []}

    if max_workers <= 1:
        for symbol in symbols:
            try:
                result = ingest_symbol(db, symbol, start_date, end_date)
                totals["inserted"] += result["inserted"]
                totals["updated"] += result["updated"]
                totals["quarantined"] += result["quarantined"]
            except Exception:
                logger.exception(f"Failed to ingest {symbol}")
                totals["failed_symbols"].append(symbol)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_ingest_symbol_isolated, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    _, result = future.result()
                    totals["inserted"] += result["inserted"]
                    totals["updated"] += result["updated"]
                    totals["quarantined"] += result["quarantined"]
                except Exception:
                    logger.exception(f"Failed to ingest {symbol}")
                    totals["failed_symbols"].append(symbol)

    logger.info(
        f"Universe ingestion complete: {totals['inserted']} inserted, "
        f"{totals['updated']} updated, {totals['quarantined']} quarantined, "
        f"{len(totals['failed_symbols'])} failed "
        f"(workers={max_workers})"
    )
    return totals
