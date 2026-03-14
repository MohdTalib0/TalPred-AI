"""Shadow-style side-by-side comparison of feature profiles."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol
from src.models.trainer import train_baseline


PROFILES = ["all_features", "liquidity_core"]


def main() -> None:
    load_dotenv()
    db = SessionLocal()
    try:
        symbols = [r.symbol for r in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]
        end = date.today()
        start = end - timedelta(days=3 * 365)

        df = build_training_dataset(
            db,
            symbols,
            start,
            end,
            target_mode="sector_relative",
            target_horizon_days=5,
            include_liquidity_features=True,
        )

        rows = []
        for profile in PROFILES:
            tr = train_baseline(
                df,
                run_name=f"shadow_profile_{profile}",
                dataset_version="v1.0-backfill",
                feature_profile=profile,
                run_mode="shadow",
            )
            bt = walk_forward_backtest(
                df,
                min_train_days=252,
                step_days=21,
                rank_top_n=20,
                rank_mode="global",
                transaction_cost_bps=10,
                rank_rebalance_stride=1,
                rank_sharpe_nw_lag=4,
                feature_profile=profile,
            )
            agg = bt["aggregate_metrics"]
            rows.append({
                "feature_profile": profile,
                "val_auc": tr["metrics"]["auc_roc"],
                "wf_auc": agg.get("overall_auc"),
                "ic_mean": agg.get("ic_mean"),
                "rolling_ic_latest_mean": agg.get("rolling_ic_latest_mean"),
                "rank_net_sharpe_nw": agg.get("rank_long_short_sharpe_net_nw"),
                "rank_net_mean": agg.get("rank_long_short_mean_net"),
                "decile_sharpe": agg.get("decile_spread_sharpe"),
                "decile_monotonicity": agg.get("decile_monotonicity_spearman"),
                "rank_mdd_net": agg.get("rank_max_drawdown_net"),
            })

        out = pd.DataFrame(rows).sort_values("rank_net_sharpe_nw", ascending=False)
        out_dir = Path("artifacts") / "research"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "shadow_profile_comparison.csv"
        out.to_csv(out_csv, index=False)

        print(f"SHADOW_PROFILE_CSV={out_csv}")
        print(out.to_string(index=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
