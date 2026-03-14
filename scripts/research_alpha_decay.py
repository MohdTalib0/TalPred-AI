"""Alpha decay analysis across prediction horizons."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol


def main() -> None:
    load_dotenv()
    db = SessionLocal()
    try:
        symbols = [r.symbol for r in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]
        end = date.today()
        start = end - timedelta(days=3 * 365)

        horizons = [1, 3, 5, 10, 20]
        rows: list[dict] = []
        for h in horizons:
            df = build_training_dataset(
                db,
                symbols,
                start,
                end,
                target_mode="sector_relative",
                target_horizon_days=h,
                include_liquidity_features=True,
            )
            bt = walk_forward_backtest(
                df,
                min_train_days=252,
                step_days=21,
                rank_top_n=20,
                rank_mode="global",
                transaction_cost_bps=10,
                rank_rebalance_stride=h,  # non-overlap by horizon for fair decay comparison
                rank_sharpe_nw_lag=max(0, h - 1),
            )
            agg = bt["aggregate_metrics"]
            rows.append({
                "horizon_days": h,
                "overall_auc": agg.get("overall_auc"),
                "ic_mean": agg.get("ic_mean"),
                "ic_ir": agg.get("ic_ir"),
                "rank_net_sharpe": agg.get("rank_long_short_sharpe_net"),
                "rank_net_sharpe_nw": agg.get("rank_long_short_sharpe_net_nw"),
                "rank_net_mean": agg.get("rank_long_short_mean_net"),
                "decile_sharpe": agg.get("decile_spread_sharpe"),
                "decile_mean": agg.get("decile_spread_mean"),
                "rank_days": agg.get("rank_days"),
            })

        out_dir = Path("artifacts") / "research"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "alpha_decay_summary.csv"
        out_df = pd.DataFrame(rows).sort_values("horizon_days")
        out_df.to_csv(out_csv, index=False)

        out_png = out_dir / "alpha_decay_summary.png"
        try:
            import matplotlib.pyplot as plt  # type: ignore

            fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            axes[0].plot(out_df["horizon_days"], out_df["ic_mean"], marker="o", label="IC Mean")
            axes[0].plot(out_df["horizon_days"], out_df["overall_auc"], marker="o", label="AUC")
            axes[0].set_ylabel("Signal Strength")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(out_df["horizon_days"], out_df["rank_net_sharpe_nw"], marker="o", label="NW Sharpe (Net)")
            axes[1].plot(out_df["horizon_days"], out_df["decile_sharpe"], marker="o", label="Decile Sharpe")
            axes[1].set_ylabel("Portfolio Quality")
            axes[1].set_xlabel("Prediction Horizon (days)")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.suptitle("Alpha Decay Across Horizons")
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.close()
            print(f"PLOT_SAVED={out_png}")
        except Exception as e:
            print(f"PLOT_SKIPPED={e}")

        print(f"ALPHA_DECAY_CSV={out_csv}")
        print(out_df.to_string(index=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
