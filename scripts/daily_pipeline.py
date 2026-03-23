"""Daily EOD pipeline orchestrator (ENG-SPEC 14).

Runs the full daily pipeline in order:
  1.  Market calendar sync
  2.  Ingest market data + data quality checks
  3.  Ingest news
  4.  Ingest macro data
  5.  Feature generation
  6.  Batch predictions
  7.  Outcome backfill (realized returns)
  8.  Monitoring checks
  9.  Paper trading monitor (subprocess — predictions + positions log)
  9b. DB simulations (legacy + strategy framework)
      Uses model_registry production model_version (not majority vote in predictions).
      SIM_FORCE_RERUN=1: delete existing simulation_runs + paper_trades for that
      date/model and re-run (use after promoting a new model).
 10.  Ingest fundamental features (Mondays only; see PIPELINE_CALENDAR_TZ / local time)
      Skipped in GitHub Actions unless FUNDAMENTALS_INGEST_IN_CI=1 (fundamentals-pipeline.yml).

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
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.ml.promotion import get_production_model  # noqa: E402
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


def _pipeline_today_weekday() -> int:
    """Weekday for pipeline 'is it Monday?' checks (0=Monday).

    If PIPELINE_CALENDAR_TZ is set (e.g. America/New_York), use it.
    Otherwise: GitHub Actions defaults to UTC; local runs use system local time.
    """
    tz_name = (os.getenv("PIPELINE_CALENDAR_TZ") or "").strip()
    if tz_name:
        try:
            return datetime.now(ZoneInfo(tz_name)).weekday()
        except Exception:
            pass
    if os.getenv("GITHUB_ACTIONS", "").strip().lower() == "true":
        return datetime.now(ZoneInfo("UTC")).weekday()
    return datetime.today().weekday()


def _fundamentals_ingest_allowed_in_ci() -> bool:
    """Heavy fundamentals ingest is off in GitHub Actions unless explicitly enabled."""
    if os.getenv("GITHUB_ACTIONS", "").strip().lower() != "true":
        return True
    return _as_bool_env("FUNDAMENTALS_INGEST_IN_CI", default=False)


def _should_run_fundamentals_today() -> bool:
    """Monday gate for fundamentals, unless FUNDAMENTALS_FORCE_RUN=1 (e.g. manual recovery)."""
    if _as_bool_env("FUNDAMENTALS_FORCE_RUN", default=False):
        return True
    return _pipeline_today_weekday() == 0


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
        db.rollback()
        logger.exception("  Calendar sync failed")


def step_2_ingest_market(db):
    """Ingest latest market bars + post-ingestion data quality checks."""
    from src.pipelines.ingest_market import ingest_universe, run_data_quality_checks
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

        dq_report = run_data_quality_checks(db, symbols, lookback_days=30)
        logger.info(
            f"  Data quality: {dq_report['overall_status']} "
            f"(gaps={len(dq_report['gap_analysis'])}, "
            f"splits={len(dq_report['split_suspects'])}, "
            f"vol_anomalies={len(dq_report['volume_anomalies'])}, "
            f"mkt_rel_outliers={len(dq_report.get('market_relative_outliers', []))})"
        )
    except Exception:
        db.rollback()
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
        db.rollback()
        logger.exception("  Feature generation failed")


def step_6_batch_predict(db):
    """Run batch predictions for all symbols."""
    from src.pipelines.batch_predict import run_batch_predictions
    logger.info("Step 6: Batch predictions + cache update")
    try:
        compute_explanations = _as_bool_env("BATCH_PREDICT_EXPLANATIONS", default=False)
        logger.info(f"  Batch predict config: compute_explanations={compute_explanations}")
        result = run_batch_predictions(
            db,
            compute_explanations=compute_explanations,
        )
        logger.info(
            f"  Predictions: {result.get('predictions', 0)}, "
            f"cached: {result.get('cached', 0)}, "
            f"errors: {result.get('errors', 0)}"
        )
    except Exception:
        db.rollback()
        logger.exception("  Batch prediction failed")


def step_7_outcome_backfill(db):
    """Backfill realized outcomes for past predictions (live IC tracking)."""
    from src.pipelines.outcome_backfill import backfill_realized_outcomes
    logger.info("Step 7: Outcome backfill (realized returns)")
    try:
        result = backfill_realized_outcomes(db, lookback_days=7)
        logger.info(
            f"  Backfilled: {result['updated']} predictions, "
            f"skipped: {result['skipped']}"
        )
    except Exception:
        db.rollback()
        logger.exception("  Outcome backfill failed")


def step_8_monitoring(db):
    """Run monitoring checks."""
    from src.monitoring.checks import run_all_checks
    logger.info("Step 8: Monitoring checks")
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
        db.rollback()
        logger.exception("  Monitoring checks failed")


def step_9_paper_trading(db):
    """Run paper trading monitor subprocess (predictions + positions log)."""
    logger.info("Step 9: Paper trading monitor")
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


def _delete_sim_runs_for_rerun(
    db,
    sim_date: date,
    model_version: str,
    strategy_names: list[str],
) -> int:
    """Remove prior runs (and trades) so simulations can be re-inserted."""
    from src.models.schema import PaperTrade, SimulationRun

    rows = (
        db.query(SimulationRun)
        .filter(
            SimulationRun.start_date == sim_date,
            SimulationRun.end_date == sim_date,
            SimulationRun.model_version == model_version,
            SimulationRun.strategy_name.in_(strategy_names),
        )
        .all()
    )
    run_ids = [r.run_id for r in rows]
    if not run_ids:
        return 0
    db.query(PaperTrade).filter(PaperTrade.run_id.in_(run_ids)).delete(
        synchronize_session=False
    )
    db.query(SimulationRun).filter(SimulationRun.run_id.in_(run_ids)).delete(
        synchronize_session=False
    )
    db.commit()
    return len(run_ids)


def step_9b_db_simulations(db):
    """Persist daily simulation runs (legacy + strategy framework) to DB."""
    from src.models.schema import SimulationRun
    from src.simulation.engine import run_simulation, run_strategy_simulation
    from src.strategies.config import StrategyFrameworkConfig
    from src.strategies.mean_reversion import MeanReversion
    from src.strategies.momentum_long_short import MomentumLongShort
    from src.strategies.momentum_reversal import MomentumReversal
    from src.strategies.sector_rotation import SectorRotation

    logger.info("Step 9b: DB simulations (legacy + strategies)")

    # ---------- Resolve production model + eligible simulation date ----------
    try:
        prod = get_production_model(db)
        if not prod or not prod.model_version:
            logger.warning(
                "  DB simulation skipped: no production model in model_registry"
            )
            return

        model_version = prod.model_version

        # Latest target_date that has both market close data and PROD model predictions
        sim_date = db.execute(
            text("""
                SELECT MAX(p.target_date)
                FROM predictions p
                WHERE p.model_version = :mv
                  AND p.target_date <= :today
                  AND EXISTS (
                      SELECT 1
                      FROM market_bars_daily mb
                      WHERE mb.date = p.target_date
                  )
            """),
            {"today": date.today(), "mv": model_version},
        ).scalar()

        if not sim_date:
            logger.info(
                "  DB simulation skipped: no eligible target_date with predictions "
                "for production model_version=%s (run batch predict after promoting)",
                model_version,
            )
            return

        logger.info(
            "  Using production model_version=%s for sim_date=%s",
            model_version,
            sim_date,
        )

        all_strategies = [
            "legacy_confidence_weighted",
            "momentum_long_short",
            "sector_rotation",
            "mean_reversion",
            "momentum_reversal",
        ]
        if _as_bool_env("SIM_FORCE_RERUN", default=False):
            n = _delete_sim_runs_for_rerun(db, sim_date, model_version, all_strategies)
            logger.info("  SIM_FORCE_RERUN: removed %d prior simulation row(s)", n)

    except Exception:
        db.rollback()
        logger.exception("  DB simulation date resolution failed")
        return

    # ---------- Legacy simulation (backward-compatible) ----------
    try:
        existing_legacy = (
            db.query(SimulationRun)
            .filter(
                SimulationRun.start_date == sim_date,
                SimulationRun.end_date == sim_date,
                SimulationRun.model_version == model_version,
                SimulationRun.strategy_name == "legacy_confidence_weighted",
                SimulationRun.status == "completed",
            )
            .first()
        )
        if existing_legacy:
            logger.info(
                f"  Legacy simulation skipped: already exists for {sim_date} "
                f"(run_id={existing_legacy.run_id})"
            )
        else:
            sim_result = run_simulation(
                db,
                start_date=sim_date,
                end_date=sim_date,
                min_confidence_trade=float(os.getenv("MIN_CONFIDENCE_TRADE", "0.65")),
                max_position=float(os.getenv("MAX_POSITION", "0.10")),
                top_n=int(os.getenv("PAPER_TOP_N", "10")),
                model_version=model_version,
            )
            if "error" in sim_result:
                logger.warning(
                    f"  Legacy simulation failed for {sim_date}: {sim_result['error']}"
                )
            else:
                m = sim_result.get("metrics", {})
                logger.info(
                    f"  DB simulation saved: run_id={sim_result['run_id']}, "
                    f"trades={sim_result.get('n_trades', 0)}, days={sim_result.get('n_trading_days', 0)}"
                )
                ic_val = m.get("ic_mean")
                ls_val = m.get("ls_spread_mean_bps")
                dec_val = m.get("decile_spread_mean_bps")
                if ic_val is not None:
                    logger.info(f"  Signal health: IC={ic_val}, LS_spread={ls_val}bps, decile_spread={dec_val}bps")
    except Exception:
        db.rollback()
        logger.exception("  Legacy simulation persistence failed")

    # ---------- Strategy framework simulations ----------
    if not _as_bool_env("STRATEGY_FRAMEWORK_ENABLED", default=True):
        logger.info("  Strategy framework disabled (STRATEGY_FRAMEWORK_ENABLED=0)")
        return

    cfg = StrategyFrameworkConfig()
    strategies = [
        MomentumLongShort(cfg.momentum),
        SectorRotation(cfg.sector_rotation),
        MeanReversion(cfg.mean_reversion),
        MomentumReversal(cfg.momentum_reversal),
    ]

    for strategy in strategies:
        try:
            existing = (
                db.query(SimulationRun)
                .filter(
                    SimulationRun.start_date == sim_date,
                    SimulationRun.end_date == sim_date,
                    SimulationRun.model_version == model_version,
                    SimulationRun.strategy_name == strategy.name,
                    SimulationRun.status == "completed",
                )
                .first()
            )
            if existing:
                logger.info(
                    f"  [{strategy.name}] skipped: already exists "
                    f"(run_id={existing.run_id})"
                )
                continue

            result = run_strategy_simulation(
                db,
                strategy=strategy,
                start_date=sim_date,
                end_date=sim_date,
                config=cfg,
                model_version=model_version,
            )
            if "error" in result:
                logger.warning(
                    f"  [{strategy.name}] failed for {sim_date}: {result['error']}"
                )
            else:
                metrics = result.get("metrics", {})
                logger.info(
                    f"  [{strategy.name}] saved: run_id={result['run_id']}, "
                    f"trades={result.get('n_trades', 0)}, "
                    f"return={metrics.get('total_return_pct', 0):.4f}%, "
                    f"alpha={metrics.get('alpha_pct', 0):.4f}%"
                )
        except Exception:
            db.rollback()
            logger.exception(f"  [{strategy.name}] simulation failed")


def step_10_ingest_fundamentals(db):
    """Ingest fundamental features (SEC EDGAR / yfinance / SimFin). Monday only."""
    if not _should_run_fundamentals_today():
        logger.info(
            "Step 10: Skipping fundamentals (not Monday — set FUNDAMENTALS_FORCE_RUN=1 to override)"
        )
        return
    if not _fundamentals_ingest_allowed_in_ci():
        logger.info(
            "Step 10: Skipping fundamentals in GitHub Actions "
            "(set FUNDAMENTALS_INGEST_IN_CI=1 or run .github/workflows/fundamentals-pipeline.yml)"
        )
        return
    from src.pipelines.ingest_fundamentals import ingest_fundamentals
    logger.info("Step 10: Ingesting fundamental features")
    try:
        symbols = _get_active_symbols(db)
        lookback = max(1, int(os.getenv("FUNDAMENTAL_LOOKBACK_YEARS", "2")))
        result = ingest_fundamentals(db, symbols, lookback_years=lookback)
        logger.info(
            f"  Fundamentals complete: {result['upserted']} upserted, "
            f"{result['symbols_covered']} symbols, {result['skipped']} skipped"
        )
    except Exception:
        db.rollback()
        logger.exception("  Fundamentals ingestion failed")


STEPS = [
    step_1_calendar_sync,       # 1
    step_2_ingest_market,       # 2
    step_3_ingest_news,         # 3
    step_4_ingest_macro,        # 4
    step_5_generate_features,   # 5
    step_6_batch_predict,       # 6
    step_7_outcome_backfill,    # 7
    step_8_monitoring,          # 8
    step_9_paper_trading,       # 9  (paper monitor subprocess)
    step_9b_db_simulations,     # 9b (legacy + strategy DB simulations)
    step_10_ingest_fundamentals,  # 10
]


def main():
    parser = argparse.ArgumentParser(description="Daily EOD pipeline")
    parser.add_argument("--step", type=int, default=1, help="Start from this step (1-10)")
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
