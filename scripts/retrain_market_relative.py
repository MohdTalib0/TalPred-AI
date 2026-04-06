"""Retrain with market-relative target and compare against absolute baseline.

Runs the full train -> walk-forward -> calibrate -> register pipeline
with target_mode="market_relative" and multi-seed robustness check.

Usage:
  python -m scripts.retrain_market_relative
  python -m scripts.retrain_market_relative --seeds 3 --allow-promote
"""

import argparse
import json
import logging
import os
import pickle
import time
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text  # noqa: E402
from src.db import SessionLocal  # noqa: E402
from src.features.leakage import build_training_dataset, validate_no_leakage  # noqa: E402
from src.ml.promotion import promote_model, register_calibration, register_model  # noqa: E402
from src.models.backtest import walk_forward_backtest  # noqa: E402
from src.models.calibration import calibrate_model  # noqa: E402
from src.features.engine import load_training_universe  # noqa: E402
from src.models.schema import Symbol  # noqa: E402
from src.models.trainer import DEFAULT_PARAMS, train_baseline, MODEL_MODES  # noqa: E402
from src.models.ensemble import train_ensemble  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)


def _run_single_seed(
    df: pd.DataFrame,
    seed: int,
    feature_profile: str,
    target_mode: str,
) -> dict:
    """Quick train/val split for a single seed. Returns metrics dict.

    Walk-forward is NOT run per-seed (too expensive: it retrains XGBoost
    at every step). Only the final chosen seed gets a full walk-forward.
    """
    params = DEFAULT_PARAMS.copy()
    params["random_state"] = seed

    from src.models.trainer import prepare_features
    import xgboost as xgb

    split_idx = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train, medians = prepare_features(df_train, feature_profile=feature_profile)
    X_val, y_val, _ = prepare_features(df_val, fill_medians=medians, feature_profile=feature_profile)

    for col in X_train.columns:
        if col not in X_val.columns:
            X_val[col] = 0
    X_val = X_val[X_train.columns]

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    from sklearn.metrics import accuracy_score, roc_auc_score
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    ic_val = float(
        pd.Series(y_prob).corr(
            pd.Series(df_val["target_value"].values[: len(y_prob)]),
            method="spearman",
        )
    ) if "target_value" in df_val.columns else None

    return {
        "seed": seed,
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "auc_roc": float(roc_auc_score(y_val, y_prob)) if len(set(y_val)) > 1 else 0.5,
        "val_ic": ic_val,
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain with market-relative target")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds for robustness")
    parser.add_argument("--feature-profile", default="cross_sectional_alpha")
    parser.add_argument("--target-horizon-days", type=int, default=5)
    parser.add_argument(
        "--sample-stride", type=int, default=0,
        help="Keep every Nth row per symbol for non-overlapping targets. "
             "0 = auto (set to target_horizon_days)."
    )
    parser.add_argument("--allow-promote", action="store_true")
    parser.add_argument("--include-liquidity-features", action="store_true")
    parser.add_argument(
        "--years", type=int, default=7,
        help="Training window in years. Default 7 covers 2019 COVID, "
             "2022 rate shock, and provides enough history for robust "
             "walk-forward evaluation. yfinance provides ~20 years free."
    )
    parser.add_argument(
        "--model-mode", default="classifier",
        choices=["classifier", "regressor", "ranker", "ensemble"],
        help="Training mode: classifier (binary), regressor (continuous return), "
             "ranker (LambdaRank pairwise), or ensemble (XGB+LGB averaged)."
    )
    parser.add_argument(
        "--include-fundamentals", action="store_true",
        help="Download SimFin fundamental data and merge earnings/accruals features. "
             "Requires SIMFIN_API_KEY environment variable."
    )
    args = parser.parse_args()

    db = SessionLocal()
    t0 = time.time()

    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.years * 365)

        # Point-in-time universe: includes stocks that were active at any point
        # during the training window, mitigating survivorship bias.
        symbols = load_training_universe(db, start_date, end_date)
        if not symbols:
            symbols = [
                row.symbol
                for row in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()
            ]
            logger.warning("PIT universe empty, falling back to currently active symbols")
        logger.info(f"Universe: {len(symbols)} symbols (survivorship-bias-aware)")
        logger.info(f"Training window: {start_date} to {end_date} ({args.years} years)")

        # Data coverage check
        coverage_result = db.execute(text("""
            SELECT MIN(date), MAX(date), COUNT(DISTINCT date)
            FROM market_bars_daily
            WHERE symbol = ANY(:syms)
        """), {"syms": symbols})
        cov = coverage_result.fetchone()
        if cov and cov[0]:
            data_start, data_end, n_dates = cov[0], cov[1], cov[2]
            logger.info(
                f"Data coverage: {data_start} to {data_end} ({n_dates} trading days)"
            )
            if data_start > start_date:
                logger.warning(
                    f"Requested {start_date} but data starts at {data_start}. "
                    f"Effective training window is shorter than {args.years} years."
                )

        stride = args.sample_stride if args.sample_stride > 0 else args.target_horizon_days

        # ── Build dataset with non-overlapping samples ──
        include_liq = True
        logger.info(
            f"Building MARKET-RELATIVE dataset "
            f"(include_liquidity={include_liq}, horizon={args.target_horizon_days}d, "
            f"sample_stride={stride}, fundamentals={args.include_fundamentals})..."
        )
        df_rel = build_training_dataset(
            db, symbols, start_date, end_date,
            target_mode="market_relative",
            include_liquidity_features=include_liq,
            target_horizon_days=args.target_horizon_days,
            sample_stride=stride,
            include_fundamentals=args.include_fundamentals,
        )

        if df_rel.empty:
            logger.error("No data. Generate features first.")
            return

        leakage = validate_no_leakage(df_rel)
        if not leakage["passed"]:
            logger.error(f"Leakage check FAILED: {leakage['violations']}")
            return

        naive_up = float(df_rel["direction"].mean())
        logger.info(f"Market-relative dataset: {len(df_rel)} rows, naive_up={naive_up:.4f}")
        logger.info(f"  (Should be ~0.50 if market factor is removed correctly)")

        # ── Seed robustness on market-relative ──
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED ROBUSTNESS CHECK ({args.seeds} seeds)")
        logger.info(f"{'='*60}")

        seed_results = []
        for seed in range(args.seeds):
            s = seed + 42
            logger.info(f"\n--- Seed {s} ---")
            result = _run_single_seed(df_rel, s, args.feature_profile, "market_relative")
            seed_results.append(result)
            logger.info(
                f"  acc={result['accuracy']:.4f}, auc={result['auc_roc']:.4f}, "
                f"val_IC={result['val_ic']}"
            )

        logger.info(f"\n{'='*60}")
        logger.info("SEED ROBUSTNESS SUMMARY (market_relative)")
        logger.info(f"{'='*60}")
        for metric in ["accuracy", "auc_roc", "val_ic"]:
            vals = [r[metric] for r in seed_results if r[metric] is not None]
            if vals:
                logger.info(
                    f"  {metric}: mean={np.mean(vals):.4f}, "
                    f"std={np.std(vals):.4f}, "
                    f"min={np.min(vals):.4f}, max={np.max(vals):.4f}"
                )

        # ── Chronological 4-way split ──
        # | Train+WF (70%) | Calibration (15%) | OOS Holdout (15%) |
        #
        # - Model trains on the first 70% (train_baseline does its own 80/20
        #   internal split within this portion).
        # - Walk-forward evaluates on the first 70%.
        # - Calibration fits on the next 15% — never seen during training.
        # - OOS holdout (last 15%) is NEVER used for anything during training,
        #   calibration, or walk-forward. It provides a truly unseen evaluation.
        cal_split = int(len(df_rel) * 0.70)
        oos_split = int(len(df_rel) * 0.85)
        df_train_wf = df_rel.iloc[:cal_split]
        df_cal = df_rel.iloc[cal_split:oos_split]
        df_oos = df_rel.iloc[oos_split:]

        logger.info(f"\n{'='*60}")
        logger.info("FULL TRAIN + REGISTER (market_relative, seed=42)")
        logger.info(f"  Training+WF rows:  {len(df_train_wf)} (first 70%)")
        logger.info(f"  Calibration rows:  {len(df_cal)} (next 15%, held out)")
        logger.info(f"  OOS holdout rows:  {len(df_oos)} (last 15%, untouched)")
        logger.info(f"{'='*60}")

        effective_mode = args.model_mode
        if effective_mode == "ensemble":
            logger.info("Training ENSEMBLE (XGBoost + LightGBM)...")
            train_result = train_ensemble(
                df_train_wf,
                feature_profile=args.feature_profile,
            )
            logger.info(f"Ensemble metrics: {train_result['metrics']}")
        else:
            train_result = train_baseline(
                df_train_wf,
                experiment_name="talpred-market-relative",
                run_name=f"train_{effective_mode}_{end_date.isoformat()}",
                feature_profile=args.feature_profile,
                run_mode="research" if not args.allow_promote else "production_candidate",
                model_mode=effective_mode,
            )
            logger.info(f"Metrics ({effective_mode}): {train_result['metrics']}")

        # Walk-forward uses classifier mode for ranking metrics even when
        # the production model is trained as regressor/ranker; this provides
        # comparable Sharpe/IC numbers. For regressor/ranker models, the WF
        # also trains with the same mode for apples-to-apples comparison.
        wf_mode = effective_mode if effective_mode in MODEL_MODES else "classifier"
        bt = walk_forward_backtest(
            df_train_wf,
            min_train_days=252,
            step_days=21,
            rank_top_n=20,
            transaction_cost_bps=10.0,
            rank_rebalance_stride=args.target_horizon_days,
            rank_sharpe_nw_lag=max(0, args.target_horizon_days - 1),
            feature_profile=args.feature_profile,
            model_mode=wf_mode,
        )
        backtest_results = bt if "error" not in bt else None
        if backtest_results:
            agg = backtest_results["aggregate_metrics"]
            logger.info(
                f"Backtest: acc={agg['overall_accuracy']:.4f}, "
                f"IC={agg.get('ic_mean')}, ICIR={agg.get('ic_ir')}, "
                f"Sharpe_net={agg.get('rank_long_short_sharpe_net')}, "
                f"MDD_net={agg.get('rank_max_drawdown_net')}, "
                f"Mono={agg.get('decile_monotonicity_spearman')}"
            )
            dsr = agg.get("rank_deflated_sharpe_ratio")
            e_max_sr = agg.get("rank_expected_max_sharpe")
            if dsr is not None:
                logger.info(
                    f"  DSR={dsr:.4f}, E[maxSR]={e_max_sr:.3f} "
                    f"(DSR>0.95 = Sharpe survives multiple testing)"
                )
                if dsr < 0.50:
                    logger.warning(
                        "  WARNING: DSR < 0.50 — Sharpe likely explained by data mining"
                    )

        # Calibrate on held-out data (never seen during training or validation)
        calibration_result = None
        if len(df_cal) > 50:
            calibration_result = calibrate_model(
                train_result["model"],
                df_cal,
                method="isotonic",
                train_medians=train_result["train_medians"],
                feature_profile=args.feature_profile,
            )
            logger.info(f"Calibration: {calibration_result['metrics']}")

        # ── OOS holdout evaluation (truly unseen data) ──
        oos_metrics: dict = {
            "oos_accuracy": None,
            "oos_auc_roc": None,
            "oos_ic": None,
            "oos_n_rows": len(df_oos),
            "oos_status": "skipped" if len(df_oos) <= 50 else "pending",
            "oos_skip_reason": (
                f"Only {len(df_oos)} OOS rows (need >50)" if len(df_oos) <= 50 else None
            ),
        }
        if len(df_oos) > 50:
            from src.models.trainer import prepare_features as _prep_feat
            from sklearn.metrics import accuracy_score as _acc, roc_auc_score as _auc

            oos_model_mode = effective_mode if effective_mode in MODEL_MODES else "classifier"
            X_oos, y_oos, _ = _prep_feat(
                df_oos, fill_medians=train_result["train_medians"],
                feature_profile=args.feature_profile,
                model_mode=oos_model_mode,
            )
            for col in train_result["feature_columns"]:
                if col not in X_oos.columns:
                    X_oos[col] = 0
            X_oos = X_oos[train_result["feature_columns"]]

            model_obj = train_result["model"]

            if oos_model_mode in ("regressor", "ranker"):
                y_scores_oos = model_obj.predict(X_oos)
                from scipy.stats import rankdata as _rankdata
                y_prob_oos = _rankdata(y_scores_oos) / len(y_scores_oos)
                y_binary_actual = (df_oos["direction"].values[:len(y_scores_oos)]
                                   if "direction" in df_oos.columns
                                   else (y_oos > 0).astype(int).values)
                y_pred_oos = (y_scores_oos > 0).astype(int)
            else:
                y_prob_oos = model_obj.predict_proba(X_oos)[:, 1]
                y_pred_oos = model_obj.predict(X_oos)
                y_binary_actual = y_oos.values

            oos_acc = float(_acc(y_binary_actual, y_pred_oos))
            oos_auc = (
                float(_auc(y_binary_actual, y_prob_oos))
                if len(set(y_binary_actual)) > 1 else 0.5
            )

            oos_ic = None
            if "target_value" in df_oos.columns:
                oos_ic = float(
                    pd.Series(y_prob_oos).corr(
                        pd.Series(df_oos["target_value"].values[:len(y_prob_oos)]),
                        method="spearman",
                    )
                )

            oos_metrics.update({
                "oos_accuracy": oos_acc,
                "oos_auc_roc": oos_auc,
                "oos_ic": oos_ic,
                "oos_n_rows": len(df_oos),
                "oos_status": "completed",
                "oos_skip_reason": None,
            })
            logger.info(f"\n{'='*60}")
            logger.info("OUT-OF-SAMPLE HOLDOUT (last 15% — truly unseen)")
            logger.info(f"{'='*60}")
            logger.info(f"  Rows:     {len(df_oos)}")
            logger.info(f"  Accuracy: {oos_acc:.4f}")
            logger.info(f"  AUC-ROC:  {oos_auc:.4f}")
            logger.info(f"  IC:       {oos_ic}")
            if oos_ic is not None and oos_ic < 0.01:
                logger.warning("  WARNING: OOS IC < 0.01 — signal may not generalize")
            if oos_acc < 0.50:
                logger.warning("  WARNING: OOS accuracy below 50% — model may be overfit")

        # Register
        training_window = train_result.get("training_window", (str(start_date), str(end_date)))

        def _pd(s, fb):
            try:
                return pd.Timestamp(s).date() if s else fb
            except Exception:
                return fb

        algo_name = {
            "classifier": "xgboost",
            "regressor": "xgboost_regressor",
            "ranker": "xgboost_ranker",
            "ensemble": "xgb_lgb_ensemble",
        }.get(effective_mode, "xgboost")

        full_metrics = {**train_result["metrics"]}

        full_metrics["oos_accuracy"] = oos_metrics.get("oos_accuracy")
        full_metrics["oos_auc_roc"] = oos_metrics.get("oos_auc_roc")
        full_metrics["oos_ic"] = oos_metrics.get("oos_ic")
        full_metrics["oos_n_rows"] = oos_metrics.get("oos_n_rows")
        full_metrics["oos_status"] = oos_metrics.get("oos_status")

        seed_accs = [r["accuracy"] for r in seed_results if r["accuracy"] is not None]
        seed_aucs = [r["auc_roc"] for r in seed_results if r["auc_roc"] is not None]
        seed_ics = [r["val_ic"] for r in seed_results if r["val_ic"] is not None]
        full_metrics["seed_accuracy_mean"] = round(float(np.mean(seed_accs)), 5) if seed_accs else None
        full_metrics["seed_accuracy_std"] = round(float(np.std(seed_accs)), 5) if seed_accs else None
        full_metrics["seed_auc_mean"] = round(float(np.mean(seed_aucs)), 5) if seed_aucs else None
        full_metrics["seed_auc_std"] = round(float(np.std(seed_aucs)), 5) if seed_aucs else None
        full_metrics["seed_ic_mean"] = round(float(np.mean(seed_ics)), 5) if seed_ics else None
        full_metrics["seed_ic_std"] = round(float(np.std(seed_ics)), 5) if seed_ics else None
        full_metrics["n_seeds"] = len(seed_results)

        if backtest_results:
            agg = backtest_results["aggregate_metrics"]
            full_metrics["wf_ic_mean"] = agg.get("ic_mean")
            full_metrics["wf_ic_ir"] = agg.get("ic_ir")
            full_metrics["wf_sharpe_net"] = agg.get("rank_long_short_sharpe_net")
            full_metrics["wf_max_drawdown"] = agg.get("rank_max_drawdown_net")
            full_metrics["wf_decile_monotonicity"] = agg.get("decile_monotonicity_spearman")
            full_metrics["wf_deflated_sharpe"] = agg.get("rank_deflated_sharpe_ratio")

        if calibration_result:
            full_metrics["cal_raw_brier"] = calibration_result["metrics"].get("raw_brier")
            full_metrics["cal_brier"] = calibration_result["metrics"].get("calibrated_brier")
            full_metrics["cal_raw_ece"] = calibration_result["metrics"].get("raw_ece")
            full_metrics["cal_ece"] = calibration_result["metrics"].get("calibrated_ece")

        full_metrics["model_mode"] = effective_mode
        full_metrics["feature_profile"] = args.feature_profile
        full_metrics["n_features"] = len(train_result["feature_columns"])
        full_metrics["target_mode"] = "market_relative"
        full_metrics["target_horizon_days"] = args.target_horizon_days
        full_metrics["training_rows"] = len(df_train_wf)
        full_metrics["include_fundamentals"] = args.include_fundamentals

        model_version = register_model(
            db,
            mlflow_run_id=train_result["run_id"],
            algorithm=algo_name,
            training_window_start=_pd(training_window[0], start_date),
            training_window_end=_pd(training_window[1], end_date),
            metrics=full_metrics,
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

        # Save artifacts
        prod_dir = os.path.join("artifacts", "production_model")
        os.makedirs(prod_dir, exist_ok=True)
        model_obj = train_result["model"]
        if hasattr(model_obj, "save_model"):
            model_obj.save_model(os.path.join(prod_dir, "model.json"))
        else:
            with open(os.path.join(prod_dir, "model_ensemble.pkl"), "wb") as f:
                pickle.dump(model_obj, f)
        with open(os.path.join(prod_dir, "train_medians.json"), "w") as f:
            json.dump(train_result["train_medians"].to_dict(), f)
        with open(os.path.join(prod_dir, "metadata.json"), "w") as f:
            json.dump({
                "model_version": model_version,
                "mlflow_run_id": train_result["run_id"],
                "model_mode": effective_mode,
                "feature_profile": args.feature_profile,
                "feature_columns": train_result["feature_columns"],
                "target_mode": "market_relative",
                "target_horizon_days": args.target_horizon_days,
                "sample_stride": stride,
                "seed_robustness": seed_results,
                "oos_holdout_metrics": oos_metrics,
                "component_models": train_result.get("component_models"),
                "saved_at": datetime.now(UTC).isoformat(),
            }, f, indent=2)

        # Promote if allowed
        if args.allow_promote:
            report = promote_model(
                db,
                model_version=model_version,
                target_status="production",
                metrics=train_result["metrics"],
                mlflow_run_id=train_result["run_id"],
                backtest_results=backtest_results,
            )
            if report["promoted"]:
                logger.info(f"Model {model_version} promoted to PRODUCTION")
            else:
                logger.warning(f"Promotion REJECTED: {report}")
        else:
            logger.info(f"Model {model_version} registered as candidate (use --allow-promote)")

        # ── Decision criteria ──
        logger.info(f"\n{'='*60}")
        logger.info("DECISION CRITERIA")
        logger.info(f"{'='*60}")
        if backtest_results:
            agg = backtest_results["aggregate_metrics"]
            ic = agg.get("ic_mean")
            icir = agg.get("ic_ir")
            mono = agg.get("decile_monotonicity_spearman")
            sharpe = agg.get("rank_long_short_sharpe_net")

            logger.info(f"  IC mean:            {ic}  (need > 0.02)")
            logger.info(f"  IC IR:              {icir}  (need > 0.5)")
            logger.info(f"  Decile monotonicity:{mono}  (need > 0.7)")
            logger.info(f"  Net Sharpe:         {sharpe}  (need > 1.0)")

            all_pass = True
            if ic is not None and ic < 0.02:
                logger.warning("  FAIL: IC too low — model has weak cross-sectional signal")
                all_pass = False
            if mono is not None and mono < 0.7:
                logger.warning("  FAIL: Decile monotonicity too low — ranking is noisy")
                all_pass = False
            if sharpe is not None and sharpe < 1.0:
                logger.warning("  WARN: Net Sharpe below 1.0 — may not survive real costs")

            if all_pass:
                logger.info("  VERDICT: Market-relative model shows real cross-sectional alpha.")
                logger.info("  Proceed with promotion and paper trading observation.")
            else:
                logger.info("  VERDICT: Weak signal. Consider feature engineering improvements.")

        elapsed = time.time() - t0
        logger.info(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    finally:
        db.close()


if __name__ == "__main__":
    main()
