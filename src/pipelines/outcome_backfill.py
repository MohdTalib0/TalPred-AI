"""Realized outcome backfill for predictions.

Looks at past predictions whose target_date has arrived, fetches the
actual market close, computes realized return and direction, and fills
the realized_return / realized_direction columns.

The return window spans from the **feature date** (as_of_time::date) to
the **target_date**, matching the model's training horizon (e.g. 5 trading
days).  Earlier versions incorrectly used a 1-day window (day-before-target
to target), which didn't match the 5-day training target.

This enables:
  - Live IC monitoring (model drift detection)
  - IC-based exposure guard in the simulation engine
  - Honest performance attribution

Respects the model's target_mode: if the model was trained on absolute
returns, backfill stores absolute returns with up/down direction.  If
market_relative, stores excess returns with outperform/underperform.
"""

import logging
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _resolve_target_mode(db: Session, model_version: str) -> str:
    """Look up the target_mode for a given model version from model_registry."""
    row = db.execute(
        text("""
            SELECT metrics->>'target_mode'
            FROM model_registry
            WHERE model_version = :v
            LIMIT 1
        """),
        {"v": model_version},
    ).fetchone()
    if row and row[0]:
        return row[0]
    return "market_relative"


def backfill_realized_outcomes(
    db: Session,
    lookback_days: int = 10,
) -> dict:
    """Fill realized_return and realized_direction for recent predictions.

    Processes predictions where:
      - target_date <= today
      - realized_return IS NULL
      - actual market data exists for target_date

    The return window is from as_of_time::date (feature date) to target_date,
    matching the model's multi-day prediction horizon.

    Returns summary dict.
    """
    result = db.execute(
        text("""
        SELECT
            p.prediction_id,
            p.symbol,
            p.target_date,
            p.direction AS predicted_direction,
            p.model_version,
            p.as_of_time::date AS feature_date,
            mb_target.close AS target_close,
            mb_base.close AS base_close,
            spy_target.close AS spy_target_close,
            spy_base.close AS spy_base_close
        FROM predictions p
        -- Close on target_date (end of horizon)
        JOIN market_bars_daily mb_target
            ON mb_target.symbol = p.symbol
            AND mb_target.date = p.target_date
        -- Close BEFORE as_of_time (the feature date close the model saw)
        LEFT JOIN LATERAL (
            SELECT close
            FROM market_bars_daily
            WHERE symbol = p.symbol
              AND date < p.as_of_time::date
            ORDER BY date DESC
            LIMIT 1
        ) mb_base ON true
        -- SPY on target_date
        LEFT JOIN market_bars_daily spy_target
            ON spy_target.symbol = 'SPY'
            AND spy_target.date = p.target_date
        -- SPY close BEFORE as_of_time (same feature date window)
        LEFT JOIN LATERAL (
            SELECT close
            FROM market_bars_daily
            WHERE symbol = 'SPY'
              AND date < p.as_of_time::date
            ORDER BY date DESC
            LIMIT 1
        ) spy_base ON true
        WHERE p.realized_return IS NULL
          AND p.target_date <= CURRENT_DATE
          AND p.target_date >= CURRENT_DATE - :lookback
        ORDER BY p.target_date
    """),
        {"lookback": lookback_days},
    )

    rows = result.fetchall()
    if not rows:
        logger.info("No predictions to backfill")
        return {"updated": 0, "skipped": 0}

    # Cache target_mode per model version to avoid repeated DB lookups
    mode_cache: dict[str, str] = {}

    updated = 0
    skipped = 0

    for row in rows:
        pred_id = row[0]
        model_version = row[4]
        feature_date = row[5]
        target_close = row[6]
        base_close = row[7]
        spy_target_close = row[8]
        spy_base_close = row[9]

        if base_close is None or base_close <= 0 or target_close is None:
            skipped += 1
            continue

        stock_return = (target_close - base_close) / base_close

        # Resolve target mode for this model version
        if model_version not in mode_cache:
            mode_cache[model_version] = _resolve_target_mode(db, model_version)
        target_mode = mode_cache[model_version]

        if target_mode in ("market_relative", "sector_relative"):
            spy_return = 0.0
            if spy_base_close and spy_base_close > 0 and spy_target_close:
                spy_return = (spy_target_close - spy_base_close) / spy_base_close

            realized_ret = stock_return - spy_return
            realized_direction = "outperform" if realized_ret > 0 else "underperform"
        else:
            realized_ret = stock_return
            realized_direction = "up" if realized_ret > 0 else "down"

        db.execute(
            text("""
            UPDATE predictions
            SET realized_return = :ret,
                realized_direction = :dir,
                outcome_recorded_at = NOW()
            WHERE prediction_id = :pid
        """),
            {
                "ret": float(realized_ret),
                "dir": realized_direction,
                "pid": pred_id,
            },
        )
        updated += 1

    db.commit()
    logger.info(
        f"Outcome backfill: {updated} updated, {skipped} skipped "
        f"(lookback={lookback_days} days)"
    )
    return {"updated": updated, "skipped": skipped}
