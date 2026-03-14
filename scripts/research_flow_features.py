"""Flow feature ablation + signal-weighted portfolio comparison.

Runs 5 controlled experiments to answer two questions:
  Q1. Do flow features (volume_acceleration, signed_volume_proxy, price_volume_trend)
      add alpha on top of the existing liquidity signal?
  Q2. Does signal-weighted portfolio construction improve Sharpe vs equal-weight?

Configs:
  A  baseline        all_features  equal-weight   (reference, matches previous best)
  B  + flow          all_features + flow  equal-weight
  C  flow_enhanced   flow_enhanced profile  equal-weight
  D  flow_only       flow_only profile      equal-weight
  E  signal_weighted all_features  signal-weight  (same model, different construction)

Usage:
  python -m scripts.research_flow_features
"""

import logging
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.trainer import train_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("research_flow_features")

# Shared experiment settings (same as previous best run)
DATASET_VERSION = "v1.0-backfill"
SHARED = dict(
    target_mode="sector_relative",
    target_horizon_days=5,
    include_liquidity_features=True,
)
BT_SHARED = dict(
    rank_top_n=20,
    rank_mode="global",
    transaction_cost_bps=10,
    rank_rebalance_stride=1,
    rank_sharpe_nw_lag=4,
)

CONFIGS = [
    {
        "name": "A_baseline",
        "feature_profile": "all_features",
        "rank_weight_mode": "equal",
        "desc": "Baseline: all_features, equal-weight (reference)",
    },
    {
        "name": "B_all_features_signal_weighted",
        "feature_profile": "all_features",
        "rank_weight_mode": "signal",
        "desc": "Signal-weighted portfolio, same all_features model",
    },
    {
        "name": "C_flow_enhanced_equal",
        "feature_profile": "flow_enhanced",
        "rank_weight_mode": "equal",
        "desc": "flow_enhanced profile (liquidity_core + flow signals), equal-weight",
    },
    {
        "name": "D_flow_enhanced_signal_weighted",
        "feature_profile": "flow_enhanced",
        "rank_weight_mode": "signal",
        "desc": "flow_enhanced profile, signal-weighted portfolio",
    },
    {
        "name": "E_flow_only",
        "feature_profile": "flow_only",
        "rank_weight_mode": "equal",
        "desc": "flow_only: size anchor + 3 flow signals only",
    },
]


def _run_config(cfg: dict, df: pd.DataFrame) -> dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {cfg['name']} — {cfg['desc']}")
    logger.info(f"{'='*60}")

    profile = cfg["feature_profile"]
    weight_mode = cfg["rank_weight_mode"]

    result = train_baseline(
        df,
        experiment_name="research-flow-features",
        run_name=cfg["name"],
        dataset_version=DATASET_VERSION,
        feature_profile=profile,
        run_mode="research",
    )
    model = result["model"]
    medians = result["train_medians"]
    feat_cols = result["feature_columns"]

    bt = walk_forward_backtest(
        df,
        params=None,
        feature_profile=profile,
        rank_weight_mode=weight_mode,
        **BT_SHARED,
    )

    agg = bt.get("aggregate_metrics", {})
    row = {
        "config": cfg["name"],
        "desc": cfg["desc"],
        "feature_profile": profile,
        "weight_mode": weight_mode,
        "n_features": len(feat_cols),
        "train_auc": result["metrics"]["auc_roc"],
        "wf_auc": agg.get("overall_auc"),
        "ic_mean": agg.get("ic_mean"),
        "ic_ir": agg.get("ic_ir"),
        "nw_sharpe_net": agg.get("rank_long_short_sharpe_net_nw"),
        "gross_sharpe": agg.get("rank_long_short_sharpe"),
        "net_spread_bps": (agg.get("rank_long_short_mean_net") or 0) * 10000,
        "max_drawdown_net": agg.get("rank_max_drawdown_net"),
        "turnover": agg.get("rank_avg_turnover"),
    }

    logger.info(f"  WF AUC:        {row['wf_auc']:.4f}" if row["wf_auc"] else "  WF AUC: N/A")
    logger.info(f"  IC mean:       {row['ic_mean']:.4f}" if row["ic_mean"] else "  IC: N/A")
    logger.info(f"  NW Sharpe net: {row['nw_sharpe_net']:.3f}" if row["nw_sharpe_net"] else "  NW Sharpe: N/A")
    logger.info(f"  Net spread:    {row['net_spread_bps']:.1f} bps/day")
    logger.info(f"  Max DD net:    {row['max_drawdown_net']:.3f}" if row["max_drawdown_net"] else "  MDD: N/A")

    return row


def main():
    from datetime import date, timedelta

    logger.info("Loading dataset...")
    db = SessionLocal()
    end_date = date.today()
    start_date = end_date - timedelta(days=3 * 365)

    from src.models.schema import Symbol
    symbols = [row.symbol for row in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]

    df = build_training_dataset(
        db,
        symbols,
        start_date,
        end_date,
        **SHARED,
    )
    db.close()

    logger.info(f"Dataset: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
                f"{df['target_session_date'].nunique()} dates")

    results = []
    for cfg in CONFIGS:
        try:
            row = _run_config(cfg, df)
            results.append(row)
        except Exception as e:
            logger.error(f"Config {cfg['name']} failed: {e}", exc_info=True)
            results.append({"config": cfg["name"], "desc": cfg["desc"], "error": str(e)})

    results_df = pd.DataFrame(results)
    logger.info("\n" + "=" * 80)
    logger.info("FLOW FEATURE ABLATION RESULTS")
    logger.info("=" * 80)

    display_cols = ["config", "weight_mode", "n_features", "wf_auc", "ic_mean", "nw_sharpe_net",
                    "net_spread_bps", "max_drawdown_net"]
    available = [c for c in display_cols if c in results_df.columns]
    logger.info("\n" + results_df[available].to_string(index=False))

    # Sort by NW Sharpe (best first)
    if "nw_sharpe_net" in results_df.columns:
        ranked = results_df.sort_values("nw_sharpe_net", ascending=False)
        logger.info("\nRANKED BY NW SHARPE:")
        logger.info(ranked[available].to_string(index=False))

    out_dir = "artifacts/research"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/flow_features_ablation.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"\nResults saved to {out_path}")

    # Key takeaway summary
    logger.info("\nKEY INTERPRETATIONS:")
    baseline = results_df[results_df["config"] == "A_baseline"]
    if not baseline.empty:
        base_sharpe = baseline.iloc[0].get("nw_sharpe_net")
        for _, row in results_df.iterrows():
            if row.get("config") == "A_baseline":
                continue
            delta = None
            if base_sharpe and row.get("nw_sharpe_net"):
                delta = row["nw_sharpe_net"] - base_sharpe
                direction = "+" if delta >= 0 else ""
                logger.info(f"  {row['config']}: NW Sharpe {direction}{delta:.3f} vs baseline "
                            f"(abs: {row['nw_sharpe_net']:.3f})")


if __name__ == "__main__":
    main()
