"""Fundamental data ingestion pipeline.

Fetches quarterly financial data from SEC EDGAR (primary, true PIT dates)
with yfinance fallback, computes fundamental features, and upserts to
the fundamental_features DB table.

Uses batch SQL INSERT ... ON CONFLICT for performance over remote DBs.
"""

import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.fundamentals import compute_fundamental_features

logger = logging.getLogger(__name__)

_FEATURE_COLS = [
    "accruals", "roe_trend", "earnings_momentum",
    "revenue_surprise", "gross_margin_change", "operating_leverage",
]

_COL_MAP: dict[str, str] = {}

_BATCH_SIZE = 500


def ingest_fundamentals(
    db: Session,
    symbols: list[str],
    lookback_years: int = 2,
) -> dict:
    """Load fundamental data, compute features, upsert to DB.

    Uses batch INSERT ... ON CONFLICT for fast remote DB writes.

    Returns {"upserted": int, "symbols_covered": int, "skipped": int}
    """
    start_date = date.today() - timedelta(days=lookback_years * 365)
    end_date = date.today()

    from src.features.fundamentals import load_fundamentals

    logger.info(
        f"Ingesting fundamentals for {len(symbols)} symbols "
        f"(lookback={lookback_years}y, {start_date} to {end_date})"
    )

    fund_data = load_fundamentals(symbols, start_date, end_date)
    features_df = compute_fundamental_features(fund_data)

    if features_df.empty:
        logger.warning("No fundamental features computed — nothing to ingest")
        return {"upserted": 0, "symbols_covered": 0, "skipped": len(symbols)}

    records = _prepare_records(features_df)
    if not records:
        return {"upserted": 0, "symbols_covered": 0, "skipped": len(symbols)}

    symbols_covered = {r["symbol"] for r in records}
    upserted = _batch_upsert(db, records)

    logger.info(
        f"Fundamentals ingestion complete: {upserted} rows upserted, "
        f"{len(symbols_covered)} symbols covered, "
        f"{len(symbols) - len(symbols_covered)} symbols skipped"
    )

    return {
        "upserted": upserted,
        "symbols_covered": len(symbols_covered),
        "skipped": len(symbols) - len(symbols_covered),
    }


def _prepare_records(features_df: pd.DataFrame) -> list[dict]:
    """Convert feature DataFrame to list of dicts for bulk upsert."""
    records = []
    for _, row in features_df.iterrows():
        symbol = row["symbol"]
        report_date = row.get("report_date")

        if report_date is None:
            continue
        if hasattr(report_date, "date"):
            try:
                report_date = report_date.date()
            except Exception:
                continue

        record = {"symbol": symbol, "as_of_date": report_date}

        for src_col in _FEATURE_COLS:
            db_col = _COL_MAP.get(src_col, src_col)
            val = row.get(src_col)
            if val is not None and not (isinstance(val, float) and val != val):
                record[db_col] = float(val)
            else:
                record[db_col] = None

        record["source"] = row.get("source", "edgar")
        if isinstance(record["source"], float) and np.isnan(record["source"]):
            record["source"] = "edgar"

        records.append(record)

    return records


def _batch_upsert(db: Session, records: list[dict]) -> int:
    """Bulk upsert using INSERT ... ON CONFLICT DO UPDATE."""
    upserted = 0
    for i in range(0, len(records), _BATCH_SIZE):
        batch = records[i:i + _BATCH_SIZE]
        for rec in batch:
            db.execute(text("""
                INSERT INTO fundamental_features
                    (symbol, as_of_date, accruals, roe_trend, earnings_momentum,
                     revenue_surprise, gross_margin_change, operating_leverage, source, updated_at)
                VALUES
                    (:symbol, :as_of_date, :accruals, :roe_trend, :earnings_momentum,
                     :revenue_surprise, :gross_margin_change, :operating_leverage, :source, NOW())
                ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                    accruals = EXCLUDED.accruals,
                    roe_trend = EXCLUDED.roe_trend,
                    earnings_momentum = EXCLUDED.earnings_momentum,
                    revenue_surprise = EXCLUDED.revenue_surprise,
                    gross_margin_change = EXCLUDED.gross_margin_change,
                    operating_leverage = EXCLUDED.operating_leverage,
                    source = EXCLUDED.source,
                    updated_at = NOW()
            """), rec)
        db.commit()
        upserted += len(batch)
        logger.info(f"  Upserted batch {i // _BATCH_SIZE + 1}: {upserted}/{len(records)} rows")

    return upserted
