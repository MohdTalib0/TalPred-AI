"""Rebalance stride comparison: 1-day vs 2-day vs 3-day.

Tests the Sharpe/cost tradeoff of rebalancing every N days for the
flow_enhanced profile on sector_relative 5-day horizon.

For a 5-day signal horizon, theoretical optimal rebalance is 1-3 days:
  stride=1  highest alpha capture,  highest turnover cost
  stride=2  balanced (hypothesis: best Sharpe/cost ratio)
  stride=3  lower cost, some alpha decay

Usage:
  python -m scripts.research_rebalance_stride
"""

import logging
import os
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol
from src.models.trainer import train_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("research_rebalance_stride")

STRIDES = [1, 2, 3]
DATASET_KW = dict(
    target_mode="sector_relative",
    target_horizon_days=5,
    include_liquidity_features=True,
)
BT_BASE = dict(
    rank_top_n=20,
    rank_mode="global",
    transaction_cost_bps=10,
    feature_profile="flow_enhanced",
)


def main():
    logger.info("Loading dataset...")
    db = SessionLocal()
    end_date = date.today()
    start_date = end_date - timedelta(days=3 * 365)
    symbols = [row.symbol for row in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]
    df = build_training_dataset(db, symbols, start_date, end_date, **DATASET_KW)
    db.close()
    logger.info(f"Dataset: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # Train once — share the same model across all stride configs
    logger.info("Training flow_enhanced model (shared across strides)...")
    train_baseline(
        df,
        experiment_name="research-rebalance-stride",
        run_name="stride_comparison",
        dataset_version="v1.0-backfill",
        feature_profile="flow_enhanced",
        run_mode="research",
    )

    results = []
    for stride in STRIDES:
        logger.info(f"\n--- Stride = {stride} day(s) ---")
        nw_lag = max(0, 5 - stride)  # NW lag scales with overlap
        bt = walk_forward_backtest(
            df,
            rank_rebalance_stride=stride,
            rank_sharpe_nw_lag=nw_lag,
            **BT_BASE,
        )
        agg = bt.get("aggregate_metrics", {})
        yearly = agg.get("rank_yearly_sharpe_net", {})

        row = {
            "stride_days": stride,
            "periods_per_year": 252 // stride,
            "wf_auc": agg.get("overall_auc"),
            "ic_mean": agg.get("ic_mean"),
            "nw_sharpe_net": agg.get("rank_long_short_sharpe_net_nw"),
            "gross_sharpe": agg.get("rank_long_short_sharpe"),
            "net_spread_bps": (agg.get("rank_long_short_mean_net") or 0) * 10000,
            "avg_daily_cost_bps": (agg.get("rank_avg_cost_daily") or 0) * 10000,
            "avg_turnover": agg.get("rank_avg_turnover"),
            "max_dd_net": agg.get("rank_max_drawdown_net"),
            "sharpe_2024": yearly.get("2024"),
            "sharpe_2025": yearly.get("2025"),
            "sharpe_2026": yearly.get("2026"),
        }
        results.append(row)

        logger.info(f"  WF AUC:        {row['wf_auc']:.4f}" if row["wf_auc"] else "  WF AUC: N/A")
        logger.info(f"  IC mean:       {row['ic_mean']:.4f}" if row["ic_mean"] else "  IC: N/A")
        logger.info(f"  NW Sharpe net: {row['nw_sharpe_net']:.3f}" if row["nw_sharpe_net"] else "  NW Sharpe: N/A")
        logger.info(f"  Net spread:    {row['net_spread_bps']:.1f} bps")
        logger.info(f"  Avg cost:      {row['avg_daily_cost_bps']:.1f} bps/day")
        logger.info(f"  Turnover:      {row['avg_turnover']:.3f}" if row["avg_turnover"] else "  Turnover: N/A")
        logger.info(f"  Max DD net:    {row['max_dd_net']:.3f}" if row["max_dd_net"] else "  MDD: N/A")

    results_df = pd.DataFrame(results)

    logger.info("\n" + "=" * 75)
    logger.info("REBALANCE STRIDE COMPARISON")
    logger.info("=" * 75)
    display = ["stride_days", "nw_sharpe_net", "net_spread_bps", "avg_daily_cost_bps",
               "avg_turnover", "max_dd_net", "sharpe_2024", "sharpe_2025"]
    available = [c for c in display if c in results_df.columns]
    logger.info("\n" + results_df[available].to_string(index=False))

    best_idx = results_df["nw_sharpe_net"].idxmax() if results_df["nw_sharpe_net"].notna().any() else None
    if best_idx is not None:
        best = results_df.iloc[best_idx]
        logger.info(f"\nBEST STRIDE: {int(best['stride_days'])} day(s) — NW Sharpe {best['nw_sharpe_net']:.3f}")
        logger.info(f"  Recommended --rank-rebalance-stride {int(best['stride_days'])}")

    out_dir = "artifacts/research"
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(f"{out_dir}/rebalance_stride_comparison.csv", index=False)
    logger.info(f"\nResults saved to {out_dir}/rebalance_stride_comparison.csv")


if __name__ == "__main__":
    main()
