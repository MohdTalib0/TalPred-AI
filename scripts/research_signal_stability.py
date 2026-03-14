"""Research utility: size control, sector breakdown, and horizon comparison."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol
from src.models.trainer import train_baseline


def _sector_alpha_breakdown(preds_df: pd.DataFrame, top_per_sector: int = 2) -> list[tuple[str, float, float | None, int]]:
    daily: list[tuple[str, pd.Timestamp, float]] = []
    for (dt, sec), sec_df in preds_df.groupby(["date", "sector"]):
        sec_df = sec_df.sort_values("probability_up", ascending=False)
        n = min(top_per_sector, len(sec_df) // 2)
        if n < 1:
            continue
        spread = float(sec_df.head(n)["target_value"].mean() - sec_df.tail(n)["target_value"].mean())
        daily.append((str(sec), pd.Timestamp(dt), spread))

    if not daily:
        return []

    ddf = pd.DataFrame(daily, columns=["sector", "date", "spread"])
    out: list[tuple[str, float, float | None, int]] = []
    for sec, grp in ddf.groupby("sector"):
        mu = float(grp["spread"].mean())
        std = float(grp["spread"].std())
        sharpe = float((mu / std) * np.sqrt(252)) if std > 0 else None
        out.append((str(sec), mu, sharpe, int(len(grp))))
    out.sort(key=lambda x: (x[2] if x[2] is not None else -999.0), reverse=True)
    return out


def main() -> None:
    load_dotenv()
    db = SessionLocal()
    try:
        symbols = [r.symbol for r in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]
        end = date.today()
        start = end - timedelta(days=3 * 365)

        print("RUNNING_SIZE_CONTROLLED_EXPERIMENTS")
        for horizon in (5, 20):
            df = build_training_dataset(
                db,
                symbols,
                start,
                end,
                target_mode="sector_relative",
                target_horizon_days=horizon,
                include_liquidity_features=True,
            )
            train_res = train_baseline(
                df,
                run_name=f"research_size_control_h{horizon}",
                dataset_version="v1.0-backfill",
            )
            importance = train_res["importance"]
            top_features = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:8]
            mcap_importance = importance.get("market_cap_rank")
            eval_configs = [("overlap", 1)]
            if horizon == 20:
                eval_configs.append(("nonoverlap", 20))

            for eval_name, stride in eval_configs:
                bt = walk_forward_backtest(
                    df,
                    min_train_days=252,
                    step_days=21,
                    rank_top_n=20,
                    rank_mode="global",
                    transaction_cost_bps=10,
                    rank_rebalance_stride=stride,
                    rank_sharpe_nw_lag=max(0, horizon - 1),
                )
                agg = bt["aggregate_metrics"]
                preds = bt["predictions"].dropna(subset=["target_value"]).copy()
                sector_stats = _sector_alpha_breakdown(preds, top_per_sector=2)

                print(
                    f"H{horizon}_{eval_name}_AUC={agg.get('overall_auc'):.4f} "
                    f"NET_SHARPE={agg.get('rank_long_short_sharpe_net'):.3f} "
                    f"NET_SHARPE_NW={agg.get('rank_long_short_sharpe_net_nw'):.3f} "
                    f"NET_SPREAD={agg.get('rank_long_short_mean_net'):.6f} "
                    f"NET_MDD={agg.get('rank_max_drawdown_net'):.3f} "
                    f"TURN={agg.get('rank_avg_turnover'):.3f} "
                    f"STRIDE={agg.get('rank_rebalance_stride')}"
                )
                if horizon == 5 and eval_name == "overlap":
                    print(f"H{horizon}_MCAP_RANK_IMPORTANCE={mcap_importance}")
                    print(f"H{horizon}_TOP_FEATURES={top_features}")
                    print(f"H{horizon}_TOP_SECTORS={sector_stats[:5]}")
                    print(f"H{horizon}_BOTTOM_SECTORS={sector_stats[-5:]}")
                if horizon == 20 and eval_name == "nonoverlap":
                    print(f"H{horizon}_NONOVERLAP_TOP_SECTORS={sector_stats[:5]}")
                    print(f"H{horizon}_NONOVERLAP_BOTTOM_SECTORS={sector_stats[-5:]}")
                print("---")
    finally:
        db.close()


if __name__ == "__main__":
    main()
