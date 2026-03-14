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
import json
import logging
import os
import pickle
import time
from datetime import UTC, date, datetime, timedelta

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
from src.models.trainer import FEATURE_PROFILES, train_baseline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train and promote model")
    parser.add_argument("--promote-to", default="production", choices=["staging", "production"])
    parser.add_argument(
        "--allow-promote",
        action="store_true",
        help="Opt-in flag required to run promotion gates and change model status.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Research mode: train/evaluate/register candidate but do not run promotion.",
    )
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="Alias for --no-promote.",
    )
    parser.add_argument("--dataset-version", type=str, default=None)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument(
        "--target-mode",
        default="absolute",
        choices=["absolute", "market_relative", "sector_relative", "market_relative_top_bottom"],
        help="Target definition for training labels.",
    )
    parser.add_argument(
        "--top-bottom-pct",
        type=float,
        default=0.30,
        help="Tail percentile for top/bottom target mode.",
    )
    parser.add_argument("--target-horizon-days", type=int, default=1, help="Prediction horizon in trading days.")
    parser.add_argument("--include-liquidity-features", action="store_true", help="Include liquidity/size features.")
    parser.add_argument("--rank-top-n", type=int, default=20, help="Top/bottom N for ranking metrics.")
    parser.add_argument(
        "--rank-mode",
        type=str,
        default="global",
        choices=["global", "sector_neutral"],
        help="Ranking construction mode for long/short metrics.",
    )
    parser.add_argument("--rank-per-sector-n", type=int, default=2, help="Top/bottom per sector when rank-mode=sector_neutral.")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0, help="Per-trade cost in basis points.")
    parser.add_argument("--rank-rebalance-stride", type=int, default=1, help="Evaluate ranking returns every Nth trading day.")
    parser.add_argument(
        "--rank-sharpe-nw-lag",
        type=int,
        default=-1,
        help="Newey-West lag for Sharpe adjustment; -1 sets lag=target_horizon_days-1.",
    )
    parser.add_argument(
        "--feature-profile",
        type=str,
        default="all_features",
        choices=sorted(FEATURE_PROFILES.keys()),
        help="Feature profile for training/inference matrix construction.",
    )
    parser.add_argument(
        "--rank-weight-mode",
        type=str,
        default="equal",
        choices=["equal", "signal"],
        help="Portfolio weighting: equal (default) or signal-weighted (proportional to |p-0.5|).",
    )
    args = parser.parse_args()
    explicit_no_promote = bool(args.no_promote or args.register_only)
    # Safe-by-default: promotion is opt-in via --allow-promote.
    no_promote = True
    if args.allow_promote and not explicit_no_promote:
        no_promote = False
    if args.allow_promote and not explicit_no_promote:
        logger.warning("PROMOTION ENABLED: production model may be updated.")

    run_mode = "research" if no_promote else "production_candidate"

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

        logger.info(
            f"Building training dataset: {start_date} -> {end_date} "
            f"(target_mode={args.target_mode})"
        )
        df = build_training_dataset(
            db,
            symbols,
            start_date,
            end_date,
            target_mode=args.target_mode,
            top_bottom_pct=args.top_bottom_pct,
            target_horizon_days=args.target_horizon_days,
            include_liquidity_features=args.include_liquidity_features,
        )

        if df.empty:
            logger.error("No training data available. Generate features first.")
            return

        logger.info(f"Dataset: {len(df)} rows, {df['symbol'].nunique()} symbols")
        naive_up_acc = float(df["direction"].mean())
        logger.info(f"Naive always-UP baseline accuracy: {naive_up_acc:.4f}")

        leakage_check = validate_no_leakage(df)
        if not leakage_check["passed"]:
            logger.error(f"Leakage check FAILED: {leakage_check['violations']}")
            return

        # ── Step 2: Train ──
        logger.info("Training baseline model...")
        train_result = train_baseline(
            df,
            experiment_name="talpred-baseline",
            run_name=f"train_{args.target_mode}_{end_date.isoformat()}",
            dataset_version=args.dataset_version,
            feature_profile=args.feature_profile,
            run_mode=run_mode,
        )
        logger.info(f"Training complete. Run ID: {train_result['run_id']}")
        logger.info(f"Metrics: {train_result['metrics']}")

        # ── Step 3: Backtest ──
        backtest_results = None
        if not args.skip_backtest:
            logger.info("Running walk-forward backtest...")
            nw_lag = args.rank_sharpe_nw_lag if args.rank_sharpe_nw_lag >= 0 else max(0, args.target_horizon_days - 1)
            backtest_results = walk_forward_backtest(
                df,
                min_train_days=252,
                step_days=21,
                rank_top_n=args.rank_top_n,
                rank_mode=args.rank_mode,
                rank_per_sector_n=args.rank_per_sector_n,
                transaction_cost_bps=args.transaction_cost_bps,
                rank_rebalance_stride=args.rank_rebalance_stride,
                rank_sharpe_nw_lag=nw_lag,
                feature_profile=args.feature_profile,
                rank_weight_mode=args.rank_weight_mode,
            )
            if "error" in backtest_results:
                logger.warning(f"Backtest issue: {backtest_results['error']}")
                backtest_results = None
            else:
                agg = backtest_results["aggregate_metrics"]
                logger.info(f"Backtest: acc={agg['overall_accuracy']:.4f}, auc={agg['overall_auc']:.4f}")
                logger.info(
                    "Ranking: mode=%s, stride=%s, nw_lag=%s, long_short_mean=%s (net=%s), sharpe=%s (net=%s), sharpe_nw=%s (net=%s), mdd=%s (net=%s), turnover=%s, cost/day=%s, days=%s",
                    agg.get("rank_mode"),
                    agg.get("rank_rebalance_stride"),
                    agg.get("rank_sharpe_nw_lag"),
                    f"{agg.get('rank_long_short_mean'):.6f}" if agg.get("rank_long_short_mean") is not None else None,
                    f"{agg.get('rank_long_short_mean_net'):.6f}" if agg.get("rank_long_short_mean_net") is not None else None,
                    f"{agg.get('rank_long_short_sharpe'):.3f}" if agg.get("rank_long_short_sharpe") is not None else None,
                    f"{agg.get('rank_long_short_sharpe_net'):.3f}" if agg.get("rank_long_short_sharpe_net") is not None else None,
                    f"{agg.get('rank_long_short_sharpe_nw'):.3f}" if agg.get("rank_long_short_sharpe_nw") is not None else None,
                    f"{agg.get('rank_long_short_sharpe_net_nw'):.3f}" if agg.get("rank_long_short_sharpe_net_nw") is not None else None,
                    f"{agg.get('rank_max_drawdown'):.3f}" if agg.get("rank_max_drawdown") is not None else None,
                    f"{agg.get('rank_max_drawdown_net'):.3f}" if agg.get("rank_max_drawdown_net") is not None else None,
                    f"{agg.get('rank_avg_turnover'):.3f}" if agg.get("rank_avg_turnover") is not None else None,
                    f"{agg.get('rank_avg_cost_daily'):.5f}" if agg.get("rank_avg_cost_daily") is not None else None,
                    agg.get("rank_days"),
                )
                logger.info(
                    "Diagnostics: IC mean=%s, IC IR=%s, rollingIC60 mean=%s, rollingIC60 IR=%s, decile_spread_mean=%s, decile_spread_sharpe=%s, decile_mdd=%s, decile_mono=%s",
                    f"{agg.get('ic_mean'):.4f}" if agg.get("ic_mean") is not None else None,
                    f"{agg.get('ic_ir'):.3f}" if agg.get("ic_ir") is not None else None,
                    f"{agg.get('rolling_ic_latest_mean'):.4f}" if agg.get("rolling_ic_latest_mean") is not None else None,
                    f"{agg.get('rolling_ic_latest_ir'):.3f}" if agg.get("rolling_ic_latest_ir") is not None else None,
                    f"{agg.get('decile_spread_mean'):.6f}" if agg.get("decile_spread_mean") is not None else None,
                    f"{agg.get('decile_spread_sharpe'):.3f}" if agg.get("decile_spread_sharpe") is not None else None,
                    f"{agg.get('decile_spread_max_drawdown'):.3f}" if agg.get("decile_spread_max_drawdown") is not None else None,
                    f"{agg.get('decile_monotonicity_spearman'):.3f}" if agg.get("decile_monotonicity_spearman") is not None else None,
                )
                logger.info(
                    "Regime diagnostics: dispersion_mean=%s, IC~dispersion=%s, IC~VIX=%s",
                    f"{agg.get('dispersion_mean'):.5f}" if agg.get("dispersion_mean") is not None else None,
                    f"{agg.get('ic_dispersion_corr'):.3f}" if agg.get("ic_dispersion_corr") is not None else None,
                    f"{agg.get('ic_vix_corr'):.3f}" if agg.get("ic_vix_corr") is not None else None,
                )
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
                    feature_profile=args.feature_profile,
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
        if no_promote:
            logger.info(
                "Promotion skipped (default-safe mode; use --allow-promote to enable). "
                f"Model {model_version} kept as candidate."
            )
            report = {
                "promoted": False,
                "skipped": True,
                "target_status": args.promote_to,
                "reason": "promotion not allowed",
            }
        else:
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

        # ── Save production artifact locally ──
        # Paper trading and batch prediction load from this path
        # instead of downloading from DagHub (which is slow/unreliable).
        prod_dir = os.path.join("artifacts", "production_model")
        os.makedirs(prod_dir, exist_ok=True)
        train_result["model"].save_model(os.path.join(prod_dir, "model.json"))
        with open(os.path.join(prod_dir, "train_medians.pkl"), "wb") as f:
            pickle.dump(train_result["train_medians"], f)
        with open(os.path.join(prod_dir, "metadata.json"), "w") as f:
            json.dump({
                "model_version": model_version,
                "mlflow_run_id": train_result["run_id"],
                "feature_profile": args.feature_profile,
                "feature_columns": train_result["feature_columns"],
                "target_mode": args.target_mode,
                "target_horizon_days": args.target_horizon_days,
                "dataset_version": args.dataset_version,
                "promoted": report.get("promoted", False),
                "saved_at": datetime.now(UTC).isoformat() if not report.get("skipped") else None,
            }, f, indent=2)
        logger.info(f"  Local artifact saved → {prod_dir}/")

        elapsed = time.time() - t0
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"  Model version: {model_version}")
        logger.info(f"  MLflow run: {train_result['run_id']}")
        if report.get("skipped"):
            logger.info(f"  Promotion skipped: {report.get('reason')}")
        else:
            logger.info(f"  Promoted to {args.promote_to}: {report['promoted']}")

        if not report.get("promoted", False) and not report.get("skipped"):
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
