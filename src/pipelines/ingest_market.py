"""Market data ingestion pipeline.

Fetches daily bars via yfinance, validates them, and upserts into market_bars_daily.
Invalid records are routed to the quarantine table.

Extended validation:
  - OHLC relationship checks (high >= low, close within [low, high])
  - Outlier detection (close price >3x or <0.33x previous close)
  - Zero-price / zero-volume guards
  - Post-ingestion gap detection across the symbol universe
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.connectors.market import fetch_daily_bars
from src.db import SessionLocal
from src.models.schema import MarketBarsDaily, QuarantineRecord

logger = logging.getLogger(__name__)

_OUTLIER_RETURN_THRESHOLD = 2.0  # 200% single-day move → quarantine


def validate_bar(row: dict, prev_close: float | None = None) -> tuple[bool, str]:
    """Validate a single market bar row.

    Args:
        row: Bar data dict.
        prev_close: Previous day's close for continuity checks.

    Returns (is_valid, reason).
    """
    required = ["symbol", "date", "open", "high", "low", "close"]
    for field in required:
        if row.get(field) is None:
            return False, f"missing required field: {field}"

    h, l, c, o = row.get("high"), row.get("low"), row["close"], row["open"]

    if h is not None and l is not None:
        if h < l:
            return False, "high < low"

    if c <= 0:
        return False, f"non-positive close: {c}"

    if o <= 0:
        return False, f"non-positive open: {o}"

    if h is not None and l is not None and c > 0:
        if c > h * 1.001 or c < l * 0.999:
            return False, f"close {c} outside [low={l}, high={h}]"

    if row.get("volume") is not None and row["volume"] < 0:
        return False, "negative volume"

    if prev_close is not None and prev_close > 0:
        ret = abs(c - prev_close) / prev_close
        if ret > _OUTLIER_RETURN_THRESHOLD:
            return False, (
                f"outlier return: {ret:.0%} move "
                f"(prev={prev_close:.2f} → close={c:.2f})"
            )

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

    # Fetch the most recent existing close for continuity checks
    prev_row = (
        db.query(MarketBarsDaily.close)
        .filter(
            MarketBarsDaily.symbol == symbol,
            MarketBarsDaily.date < start_date,
        )
        .order_by(MarketBarsDaily.date.desc())
        .first()
    )
    prev_close = float(prev_row[0]) if prev_row else None

    inserted = 0
    updated = 0
    quarantined = 0

    df_sorted = df.sort_values("date")
    for _, row in df_sorted.iterrows():
        record = row.to_dict()
        is_valid, reason = validate_bar(record, prev_close=prev_close)

        if not is_valid:
            db.add(QuarantineRecord(
                source="market_bars_daily",
                record_data={
                    k: (str(v) if isinstance(v, (date, datetime)) else v)
                    for k, v in record.items()
                },
                failure_reason=reason,
            ))
            quarantined += 1
            continue

        # Update prev_close for the next row's continuity check
        prev_close = record["close"]

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


# ------------------------------------------------------------------
# Post-ingestion data quality checks
# ------------------------------------------------------------------

def run_data_quality_checks(
    db: Session,
    symbols: list[str],
    lookback_days: int = 30,
) -> dict:
    """Run comprehensive data quality checks on recently ingested data.

    Returns a report dict with:
      - gap_analysis: symbols with missing trading days
      - split_suspects: symbols with suspicious adjustment jumps
      - volume_anomalies: symbols with zero-volume streaks
      - quarantine_summary: recent quarantine entries
    """
    report: dict = {
        "gap_analysis": [],
        "split_suspects": [],
        "volume_anomalies": [],
        "quarantine_summary": {},
        "overall_status": "ok",
    }

    report["gap_analysis"] = _detect_gaps(db, symbols, lookback_days)
    report["split_suspects"] = _detect_split_anomalies(db, symbols, lookback_days)
    report["volume_anomalies"] = _detect_volume_anomalies(db, symbols, lookback_days)
    report["market_relative_outliers"] = _detect_market_relative_outliers(
        db, symbols, lookback_days,
    )
    report["quarantine_summary"] = _quarantine_stats(db, lookback_days)

    issues = (
        len(report["gap_analysis"])
        + len(report["split_suspects"])
        + len(report["volume_anomalies"])
        + len(report["market_relative_outliers"])
    )
    if issues > 0:
        report["overall_status"] = "warning"
        logger.warning(f"Data quality check found {issues} issues across universe")
    else:
        logger.info("Data quality checks passed — no issues found")

    return report


def _detect_gaps(
    db: Session,
    symbols: list[str],
    lookback_days: int,
) -> list[dict]:
    """Find symbols with missing trading days in recent data.

    Uses SPY as a trading calendar reference. A symbol is flagged if it
    has fewer than 80% of SPY's trading days in the lookback window.
    """
    result = db.execute(text("""
        WITH calendar AS (
            SELECT date FROM market_bars_daily
            WHERE symbol = 'SPY'
              AND date >= CURRENT_DATE - :lookback
        ),
        coverage AS (
            SELECT
                mb.symbol,
                COUNT(DISTINCT mb.date) AS bar_count,
                (SELECT COUNT(*) FROM calendar) AS expected_count
            FROM market_bars_daily mb
            WHERE mb.symbol = ANY(:syms)
              AND mb.date >= CURRENT_DATE - :lookback
            GROUP BY mb.symbol
        )
        SELECT symbol, bar_count, expected_count
        FROM coverage
        WHERE bar_count < expected_count * 0.8
        ORDER BY bar_count ASC
    """), {"syms": symbols, "lookback": lookback_days})

    gaps = []
    for row in result:
        coverage_pct = round(row[1] / max(row[2], 1) * 100, 1)
        gaps.append({
            "symbol": row[0],
            "bars_found": row[1],
            "bars_expected": row[2],
            "coverage_pct": coverage_pct,
        })
        logger.warning(
            "%s: only %d/%d trading days (%.1f%% coverage)",
            row[0], row[1], row[2], coverage_pct,
        )

    return gaps


def _detect_split_anomalies(
    db: Session,
    symbols: list[str],
    lookback_days: int,
) -> list[dict]:
    """Detect adj_close/close ratio jumps that suggest bad split handling."""
    result = db.execute(text("""
        WITH ratios AS (
            SELECT
                symbol, date,
                close, adj_close,
                CASE WHEN close > 0 THEN adj_close / close ELSE NULL END AS adj_ratio,
                LAG(CASE WHEN close > 0 THEN adj_close / close ELSE NULL END)
                    OVER (PARTITION BY symbol ORDER BY date) AS prev_ratio
            FROM market_bars_daily
            WHERE symbol = ANY(:syms)
              AND date >= CURRENT_DATE - :lookback
        )
        SELECT symbol, date, adj_ratio, prev_ratio,
               ABS(adj_ratio - prev_ratio) AS ratio_shift
        FROM ratios
        WHERE prev_ratio IS NOT NULL
          AND adj_ratio IS NOT NULL
          AND ABS(adj_ratio - prev_ratio) > 0.05
        ORDER BY ratio_shift DESC
        LIMIT 50
    """), {"syms": symbols, "lookback": lookback_days})

    suspects = []
    for row in result:
        suspects.append({
            "symbol": row[0],
            "date": str(row[1]),
            "adj_ratio": round(float(row[2]), 4),
            "prev_ratio": round(float(row[3]), 4),
            "shift": round(float(row[4]), 4),
        })

    if suspects:
        logger.warning(
            "%d split/adjustment anomalies found in last %d days",
            len(suspects), lookback_days,
        )

    return suspects


def _detect_volume_anomalies(
    db: Session,
    symbols: list[str],
    lookback_days: int,
) -> list[dict]:
    """Find symbols with extended zero-volume streaks (possible delisted/halted)."""
    result = db.execute(text("""
        SELECT symbol, COUNT(*) AS zero_vol_days
        FROM market_bars_daily
        WHERE symbol = ANY(:syms)
          AND date >= CURRENT_DATE - :lookback
          AND (volume = 0 OR volume IS NULL)
        GROUP BY symbol
        HAVING COUNT(*) >= 3
        ORDER BY zero_vol_days DESC
    """), {"syms": symbols, "lookback": lookback_days})

    anomalies = []
    for row in result:
        anomalies.append({
            "symbol": row[0],
            "zero_volume_days": row[1],
        })
        logger.warning(
            "%s: %d zero-volume days in last %d days",
            row[0], row[1], lookback_days,
        )

    return anomalies


def _detect_market_relative_outliers(
    db: Session,
    symbols: list[str],
    lookback_days: int,
    stock_threshold: float = 0.15,
    spy_threshold: float = 0.05,
) -> list[dict]:
    """Flag bars where a stock moved >15% but SPY moved <5%.

    A large stock-specific move without a corresponding market move
    strongly suggests unadjusted split/reverse-split data from yfinance.
    These bars are logged for manual review (not auto-quarantined, since
    some are legitimate — e.g. biotech FDA decisions).
    """
    result = db.execute(text("""
        WITH stock_returns AS (
            SELECT
                symbol, date, close,
                LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
            FROM market_bars_daily
            WHERE symbol = ANY(:syms)
              AND date >= CURRENT_DATE - :lookback
        ),
        spy_returns AS (
            SELECT
                date, close,
                LAG(close) OVER (ORDER BY date) AS prev_close
            FROM market_bars_daily
            WHERE symbol = 'SPY'
              AND date >= CURRENT_DATE - :lookback
        )
        SELECT
            s.symbol, s.date,
            ABS((s.close - s.prev_close) / NULLIF(s.prev_close, 0)) AS stock_ret,
            ABS((spy.close - spy.prev_close) / NULLIF(spy.prev_close, 0)) AS spy_ret
        FROM stock_returns s
        JOIN spy_returns spy ON s.date = spy.date
        WHERE s.prev_close > 0
          AND spy.prev_close > 0
          AND ABS((s.close - s.prev_close) / s.prev_close) > :stock_thresh
          AND ABS((spy.close - spy.prev_close) / spy.prev_close) < :spy_thresh
        ORDER BY stock_ret DESC
        LIMIT 100
    """), {
        "syms": symbols,
        "lookback": lookback_days,
        "stock_thresh": stock_threshold,
        "spy_thresh": spy_threshold,
    })

    outliers = []
    for row in result:
        outliers.append({
            "symbol": row[0],
            "date": str(row[1]),
            "stock_return_abs": round(float(row[2]) * 100, 1),
            "spy_return_abs": round(float(row[3]) * 100, 1),
        })
        logger.warning(
            "%s on %s: %.1f%% move while SPY moved only %.1f%% — "
            "possible unadjusted split",
            row[0], row[1],
            float(row[2]) * 100,
            float(row[3]) * 100,
        )

    return outliers


def _quarantine_stats(db: Session, lookback_days: int) -> dict:
    """Summarize recent quarantine entries by failure reason."""
    result = db.execute(text("""
        SELECT failure_reason, COUNT(*) as cnt
        FROM quarantine_records
        WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day' * :lookback
          AND source = 'market_bars_daily'
        GROUP BY failure_reason
        ORDER BY cnt DESC
    """), {"lookback": lookback_days})

    stats = {}
    for row in result:
        stats[row[0]] = row[1]

    if stats:
        total = sum(stats.values())
        logger.info(f"Quarantine: {total} records in last {lookback_days} days")
        for reason, cnt in stats.items():
            logger.info(f"  {reason}: {cnt}")

    return stats
