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
import subprocess
import sys
import time
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.models.schema import Symbol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("daily_pipeline")


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
        result = ingest_universe(db, symbols, today - timedelta(days=5), today)
        logger.info(f"  Market ingestion complete: {result}")
    except Exception:
        logger.exception("  Market ingestion failed")


def step_3_ingest_news(db):
    """Ingest latest news events."""
    from src.pipelines.ingest_news import ingest_news
    logger.info("Step 3: News ingestion")
    try:
        symbols = _get_active_symbols(db)
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
    """Generate features for latest session (incremental)."""
    from src.features.engine import generate_features, save_snapshots
    logger.info("Step 5: Feature generation")
    try:
        symbols = _get_active_symbols(db)
        snapshots = generate_features(db, symbols, target_dates=None)
        saved = save_snapshots(db, snapshots)
        logger.info(f"  Features generated: {saved} snapshots")
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
    """Run paper trading monitor (uses production model artifact)."""
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
