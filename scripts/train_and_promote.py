"""End-to-end training, evaluation, and promotion pipeline.

Orchestrates:
1. Build training dataset from features
2. Train baseline XGBoost
3. Run walk-forward backtest
4. Calibrate probabilities
5. Register in model_registry
6. Run promotion gates
7. If gates pass, promote to production

Usage:
  python -m scripts.train_and_promote
  python -m scripts.train_and_promote --promote-to staging
  python -m scripts.train_and_promote --dataset-version dvc:abc123
"""

import argparse
import logging
import time
from datetime import date, timedelta

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.features.leakage import build_training_dataset, validate_no_leakage  # noqa: E402
from src.ml.promotion import (  # noqa: E402
    promote_model,
    register_calibration,
    register_model,
)
from src.models.backtest import walk_forward_backtest  # noqa: E402
from src.models.calibration import calibrate_model  # noqa: E402
from src.models.schema import Symbol  # noqa: E402
from src.models.trainer import train_baseline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train and promote model")
    parser.add_argument("--promote-to", default="production", choices=["staging", "production"])
    parser.add_argument("--dataset-version", type=str, default=None)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    db = SessionLocal()
    t0 = time.time()

    try:
        symbols = [
            row.symbol for row in
            db.query(Symbol).filter(Symbol.is_active.is_(True)).all()
        ]
        logger.info(f"Universe: {len(symbols)} active symbols")

        end_date = date.today()
        start_date = end_date - timedelta(days=3 * 365)

        logger.info(f"Building training dataset: {start_date} -> {end_date}")
        df = build_training_dataset(db, symbols, start_date, end_date)

        if df.empty:
            logger.error("No training data available. Generate features first.")
            return

        logger.info(f"Dataset: {len(df)} rows, {df['symbol'].nunique()} symbols")

        leakage_check = validate_no_leakage(df)
        if not leakage_check["passed"]:
            logger.error(f"Leakage check FAILED: {leakage_check['violations']}")
            return

        # ── Step 2: Train ──
        logger.info("Training baseline model...")
        train_result = train_baseline(
            df,
            experiment_name="talpred-baseline",
            run_name=f"train_{end_date.isoformat()}",
            dataset_version=args.dataset_version,
        )
        logger.info(f"Training complete. Run ID: {train_result['run_id']}")
        logger.info(f"Metrics: {train_result['metrics']}")

        # ── Step 3: Backtest ──
        backtest_results = None
        if not args.skip_backtest:
            logger.info("Running walk-forward backtest...")
            backtest_results = walk_forward_backtest(df, min_train_days=252, step_days=21)
            if "error" in backtest_results:
                logger.warning(f"Backtest issue: {backtest_results['error']}")
                backtest_results = None
            else:
                agg = backtest_results["aggregate_metrics"]
                logger.info(f"Backtest: acc={agg['overall_accuracy']:.4f}, auc={agg['overall_auc']:.4f}")
        else:
            logger.info("Backtest skipped")

        # ── Step 4: Calibrate ──
        calibration_result = None
        if not args.skip_calibration:
            cal_split = int(len(df) * 0.85)
            df_cal = df.iloc[cal_split:]
            if len(df_cal) > 50:
                logger.info(f"Calibrating on {len(df_cal)} rows...")
                calibration_result = calibrate_model(
                    train_result["model"],
                    df_cal,
                    method="isotonic",
                    train_medians=train_result["train_medians"],
                )
                logger.info(f"Calibration: {calibration_result['metrics']}")
            else:
                logger.warning("Not enough data for calibration")
        else:
            logger.info("Calibration skipped")

        # ── Step 5: Register ──
        training_window = train_result.get("training_window", (str(start_date), str(end_date)))

        def _parse_date(s: str | None, fallback: date) -> date:
            if not s:
                return fallback
            try:
                return pd.Timestamp(s).date()
            except Exception:
                return fallback

        model_version = register_model(
            db,
            mlflow_run_id=train_result["run_id"],
            algorithm="xgboost",
            training_window_start=_parse_date(training_window[0], start_date),
            training_window_end=_parse_date(training_window[1], end_date),
            metrics=train_result["metrics"],
            dataset_version=args.dataset_version,
            status="candidate",
        )

        if calibration_result:
            register_calibration(
                db,
                model_version=model_version,
                calibration_type=calibration_result["method"],
                training_window=f"{start_date} to {end_date}",
                calibration_metrics=calibration_result["metrics"],
            )

        # ── Step 6: Promote ──
        logger.info(f"Running promotion gates for {model_version}...")
        report = promote_model(
            db,
            model_version=model_version,
            target_status=args.promote_to,
            metrics=train_result["metrics"],
            mlflow_run_id=train_result["run_id"],
            dataset_version=args.dataset_version,
            backtest_results=backtest_results,
        )

        elapsed = time.time() - t0
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"  Model version: {model_version}")
        logger.info(f"  MLflow run: {train_result['run_id']}")
        logger.info(f"  Promoted to {args.promote_to}: {report['promoted']}")

        if not report["promoted"]:
            logger.warning("  Promotion REJECTED. Gate details:")
            for gate_name in ["kpi_gates", "lineage_check", "backtest_check"]:
                gate = report.get(gate_name, {})
                logger.warning(f"    {gate_name}: {gate}")

        logger.info(f"  Accuracy: {train_result['metrics']['accuracy']:.4f}")
        logger.info(f"  AUC-ROC: {train_result['metrics']['auc_roc']:.4f}")
        logger.info(f"{'='*60}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
