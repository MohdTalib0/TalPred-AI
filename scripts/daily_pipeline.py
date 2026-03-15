"""Daily EOD pipeline orchestrator (ENG-SPEC 14).

Runs the full daily pipeline in order:
1. Market calendar sync
2. Ingest market data
3. Ingest news
4. Ingest macro data
5. Feature generation
6. Batch predictions
7. Redis cache update (part of batch predict)
8. Monitoring checks

Usage:
  python -m scripts.daily_pipeline              # run full pipeline
  python -m scripts.daily_pipeline --step 6     # run from step 6 onwards
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import date, timedelta

from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.models.schema import Symbol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("daily_pipeline")


def _as_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_active_symbols(db) -> list[str]:
    return [
        row.symbol for row in
        db.query(Symbol).filter(Symbol.is_active.is_(True)).all()
    ]


def step_1_calendar_sync(db):
    """Sync market calendar for upcoming sessions."""
    from src.calendar.service import sync_calendar_to_db
    logger.info("Step 1: Calendar sync")
    try:
        today = date.today()
        count = sync_calendar_to_db(db, today - timedelta(days=7), today + timedelta(days=30))
        logger.info(f"  Calendar sync complete: {count} entries")
    except Exception:
        logger.exception("  Calendar sync failed")


def step_2_ingest_market(db):
    """Ingest latest market bars."""
    from src.pipelines.ingest_market import ingest_universe
    logger.info("Step 2: Market data ingestion")
    try:
        symbols = _get_active_symbols(db)
        today = date.today()
        workers = max(1, int(os.getenv("MARKET_INGEST_WORKERS", "8")))
        lookback_days = max(1, int(os.getenv("MARKET_INGEST_LOOKBACK_DAYS", "5")))
        logger.info(f"  Market ingest config: workers={workers}, lookback_days={lookback_days}")
        result = ingest_universe(
            db,
            symbols,
            today - timedelta(days=lookback_days),
            today,
            max_workers=workers,
        )
        logger.info(f"  Market ingestion complete: {result}")
    except Exception:
        logger.exception("  Market ingestion failed")


def step_3_ingest_news(db):
    """Ingest latest news events."""
    from src.pipelines.ingest_news import ingest_news
    logger.info("Step 3: News ingestion")
    try:
        if not _as_bool_env("NEWS_INGEST_ENABLED", default=False):
            logger.info("  News ingestion skipped (NEWS_INGEST_ENABLED=false)")
            return
        symbols = _get_active_symbols(db)
        symbol_limit = max(1, int(os.getenv("NEWS_SYMBOL_LIMIT", "150")))
        if symbol_limit < len(symbols):
            symbols = symbols[:symbol_limit]
        logger.info(f"  News ingest config: symbols={len(symbols)}")
        today = date.today()
        result = ingest_news(db, symbols, today - timedelta(days=2), today)
        logger.info(f"  News ingestion complete: {result}")
    except Exception:
        logger.exception("  News ingestion failed")


def step_4_ingest_macro(db):
    """Ingest latest macro data."""
    from src.pipelines.ingest_macro import ingest_macro
    logger.info("Step 4: Macro data ingestion")
    try:
        today = date.today()
        result = ingest_macro(db, today - timedelta(days=30), today)
        logger.info(f"  Macro ingestion complete: {result}")
    except Exception:
        logger.exception("  Macro ingestion failed")


def step_5_generate_features(db):
    """Generate features for latest session and persist sector returns."""
    from src.features.engine import (
        compute_sector_returns,
        generate_features,
        load_market_window,
        load_sector_map,
        save_sector_returns,
        save_snapshots,
    )
    logger.info("Step 5: Feature generation")
    try:
        symbols = _get_active_symbols(db)
        snapshots = generate_features(db, symbols, target_dates=None)
        saved = save_snapshots(db, snapshots)
        logger.info(f"  Features generated: {saved} snapshots")

        # Persist latest sector returns daily so monitoring/audits can query DB.
        latest_bar = db.execute(text("SELECT MAX(date) FROM market_bars_daily")).scalar()
        if latest_bar:
            lookback_days = max(30, int(os.getenv("SECTOR_RETURNS_LOOKBACK_DAYS", "120")))
            market_df = load_market_window(db, symbols, lookback_days=lookback_days)
            sector_map = load_sector_map(db)
            sector_df = compute_sector_returns(market_df, sector_map)
            if not sector_df.empty:
                sector_df = sector_df[sector_df["date"].dt.date == latest_bar]
                saved_sector = save_sector_returns(db, sector_df)
                logger.info(
                    f"  Sector returns saved: {saved_sector} rows (date={latest_bar})"
                )
            else:
                logger.info("  Sector returns skipped: no sector return rows generated")
        else:
            logger.info("  Sector returns skipped: no market bars available")
    except Exception:
        logger.exception("  Feature generation failed")


def step_6_batch_predict(db):
    """Run batch predictions for all symbols."""
    from src.pipelines.batch_predict import run_batch_predictions
    logger.info("Step 6: Batch predictions + cache update")
    try:
        result = run_batch_predictions(db, compute_explanations=True)
        logger.info(
            f"  Predictions: {result.get('predictions', 0)}, "
            f"cached: {result.get('cached', 0)}, "
            f"errors: {result.get('errors', 0)}"
        )
    except Exception:
        logger.exception("  Batch prediction failed")


def step_7_monitoring(db):
    """Run monitoring checks."""
    from src.monitoring.checks import run_all_checks
    logger.info("Step 7: Monitoring checks")
    try:
        report = run_all_checks(db)
        logger.info(
            f"  Status: {report['overall_status']}, "
            f"alerts: {report['total_alerts']}"
        )
        if report["total_alerts"] > 0:
            for alert in report["alert_details"]:
                logger.warning(f"  ALERT [{alert['level']}] {alert['check']}: {alert['detail']}")
    except Exception:
        logger.exception("  Monitoring checks failed")


def step_8_paper_trading(db):
    """Run paper monitor and persist daily simulation run/trades to DB."""
    from src.models.schema import SimulationRun
    from src.simulation.engine import run_simulation

    logger.info("Step 8: Paper trading monitor")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.paper_trading_monitor"],
            capture_output=True,
            text=True,
            timeout=180,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                logger.info(f"  {line}")
        if result.returncode != 0:
            logger.warning(f"  Paper trading exited with code {result.returncode}")
            for line in result.stderr.strip().split("\n")[-10:]:
                if line.strip():
                    logger.warning(f"  {line}")
    except subprocess.TimeoutExpired:
        logger.warning("  Paper trading monitor timed out (180s)")
    except Exception:
        logger.exception("  Paper trading monitor failed")

    # Persist DB-native simulation/paper-trade records for daily tracking.
    try:
        sim_date = db.execute(
            text("""
                SELECT MAX(p.target_date)
                FROM predictions p
                WHERE p.target_date <= :today
                  AND EXISTS (
                      SELECT 1
                      FROM market_bars_daily mb
                      WHERE mb.date = p.target_date
                  )
            """),
            {"today": date.today()},
        ).scalar()

        if not sim_date:
            logger.info("  DB simulation skipped: no eligible target_date available")
            return

        model_version = db.execute(
            text("""
                SELECT model_version
                FROM predictions
                WHERE target_date = :td
                GROUP BY model_version
                ORDER BY COUNT(*) DESC, MAX(as_of_time) DESC
                LIMIT 1
            """),
            {"td": sim_date},
        ).scalar()

        existing = (
            db.query(SimulationRun)
            .filter(
                SimulationRun.start_date == sim_date,
                SimulationRun.end_date == sim_date,
                SimulationRun.model_version == model_version,
                SimulationRun.status == "completed",
            )
            .first()
        )
        if existing:
            logger.info(
                f"  DB simulation skipped: already exists for {sim_date} "
                f"(run_id={existing.run_id})"
            )
            return

        sim_result = run_simulation(
            db,
            start_date=sim_date,
            end_date=sim_date,
            min_confidence_trade=float(os.getenv("MIN_CONFIDENCE_TRADE", "0.60")),
            max_position=float(os.getenv("MAX_POSITION", "0.05")),
            top_n=int(os.getenv("PAPER_TOP_N", "20")),
            model_version=model_version,
        )
        if "error" in sim_result:
            logger.warning(
                f"  DB simulation failed for {sim_date}: {sim_result['error']}"
            )
        else:
            logger.info(
                f"  DB simulation saved: run_id={sim_result['run_id']}, "
                f"trades={sim_result.get('n_trades', 0)}, days={sim_result.get('n_trading_days', 0)}"
            )
    except Exception:
        logger.exception("  DB simulation persistence failed")


STEPS = [
    step_1_calendar_sync,
    step_2_ingest_market,
    step_3_ingest_news,
    step_4_ingest_macro,
    step_5_generate_features,
    step_6_batch_predict,
    step_7_monitoring,
    step_8_paper_trading,
]


def main():
    parser = argparse.ArgumentParser(description="Daily EOD pipeline")
    parser.add_argument("--step", type=int, default=1, help="Start from this step (1-8)")
    args = parser.parse_args()

    db = SessionLocal()
    t0 = time.time()

    logger.info(f"{'='*60}")
    logger.info(f"Daily pipeline starting (date={date.today()}, from step {args.step})")
    logger.info(f"{'='*60}")

    try:
        for i, step_fn in enumerate(STEPS, 1):
            if i < args.step:
                continue

            step_t0 = time.time()
            step_fn(db)
            logger.info(f"  Step {i} took {time.time() - step_t0:.1f}s")

    finally:
        db.close()

    elapsed = time.time() - t0
    logger.info(f"{'='*60}")
    logger.info(f"Daily pipeline complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
