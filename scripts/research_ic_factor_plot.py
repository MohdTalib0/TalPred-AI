"""Generate IC diagnostics and cumulative decile-factor return plot."""

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

        df = build_training_dataset(
            db,
            symbols,
            start,
            end,
            target_mode="sector_relative",
            target_horizon_days=5,
            include_liquidity_features=True,
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
        )
        agg = bt["aggregate_metrics"]
        curve = agg.get("decile_curve", [])
        if not curve:
            print("No decile curve generated.")
            return

        out_dir = Path("artifacts") / "research"
        out_dir.mkdir(parents=True, exist_ok=True)

        curve_df = pd.DataFrame(curve)
        curve_df["date"] = pd.to_datetime(curve_df["date"])
        csv_path = out_dir / "h5_decile_curve.csv"
        curve_df.to_csv(csv_path, index=False)

        rolling_ic_curve = agg.get("rolling_ic_curve", [])
        rolling_ic_path = out_dir / "h5_rolling_ic_curve.csv"
        if rolling_ic_curve:
            ric_df = pd.DataFrame(rolling_ic_curve)
            ric_df["date"] = pd.to_datetime(ric_df["date"])
            ric_df.to_csv(rolling_ic_path, index=False)
        else:
            ric_df = pd.DataFrame()

        prob_bins = agg.get("probability_return_bins", [])
        prob_bins_path = out_dir / "h5_probability_bins.csv"
        if prob_bins:
            pd.DataFrame(prob_bins).to_csv(prob_bins_path, index=False)

        triplet_curve = agg.get("ic_vix_dispersion_curve", [])
        triplet_path = out_dir / "h5_ic_vix_dispersion_curve.csv"
        if triplet_curve:
            triplet_df = pd.DataFrame(triplet_curve)
            triplet_df["date"] = pd.to_datetime(triplet_df["date"])
            triplet_df.to_csv(triplet_path, index=False)
        else:
            triplet_df = pd.DataFrame()

        regime_ic = agg.get("regime_ic", {})
        regime_ic_path = out_dir / "h5_regime_ic.json"
        if regime_ic:
            import json
            regime_ic_path.write_text(json.dumps(regime_ic, indent=2), encoding="utf-8")

        vix_ic = agg.get("vix_bucket_ic", {})
        vix_ic_path = out_dir / "h5_vix_bucket_ic.json"
        if vix_ic:
            import json
            vix_ic_path.write_text(json.dumps(vix_ic, indent=2), encoding="utf-8")

        png_path = out_dir / "h5_decile_curve.png"
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(10, 5))
            plt.plot(curve_df["date"], curve_df["cum_decile_spread"], linewidth=1.6)
            plt.title("H5 Top-Decile Minus Bottom-Decile Cumulative Return")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return Index")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=160)
            plt.close()
            print(f"PLOT_SAVED={png_path}")

            if not ric_df.empty:
                ric_png = out_dir / "h5_rolling_ic.png"
                plt.figure(figsize=(10, 5))
                plt.plot(ric_df["date"], ric_df["rolling_ic_mean"], linewidth=1.6, label="Rolling IC Mean (60d)")
                if "rolling_ic_ir" in ric_df.columns:
                    plt.plot(ric_df["date"], ric_df["rolling_ic_ir"], linewidth=1.2, alpha=0.8, label="Rolling IC IR (60d)")
                plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
                plt.title("H5 Rolling IC Diagnostics")
                plt.xlabel("Date")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(ric_png, dpi=160)
                plt.close()
                print(f"PLOT_SAVED={ric_png}")

            if not triplet_df.empty:
                triplet_png = out_dir / "h5_ic_vix_dispersion.png"
                fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                axes[0].plot(triplet_df["date"], triplet_df["ic"], linewidth=1.4)
                axes[0].set_ylabel("Daily IC")
                axes[0].grid(True, alpha=0.3)
                axes[1].plot(triplet_df["date"], triplet_df["vix"], linewidth=1.2, color="tab:orange")
                axes[1].set_ylabel("VIX")
                axes[1].grid(True, alpha=0.3)
                axes[2].plot(triplet_df["date"], triplet_df["dispersion"], linewidth=1.2, color="tab:green")
                axes[2].set_ylabel("Cross-sec Disp")
                axes[2].grid(True, alpha=0.3)
                plt.suptitle("IC vs VIX vs Cross-Sectional Dispersion (H5)")
                plt.tight_layout()
                plt.savefig(triplet_png, dpi=160)
                plt.close()
                print(f"PLOT_SAVED={triplet_png}")
        except Exception as e:
            print(f"PLOT_SKIPPED={e}")

        print(
            "IC_AND_DECILE_SUMMARY "
            f"IC_MEAN={agg.get('ic_mean'):.4f} "
            f"IC_IR={agg.get('ic_ir'):.3f} "
            f"ROLLING_IC_60_MEAN={agg.get('rolling_ic_latest_mean') if agg.get('rolling_ic_latest_mean') is not None else None} "
            f"ROLLING_IC_60_IR={agg.get('rolling_ic_latest_ir') if agg.get('rolling_ic_latest_ir') is not None else None} "
            f"DECILE_SHARPE={agg.get('decile_spread_sharpe'):.3f} "
            f"DECILE_MDD={agg.get('decile_spread_max_drawdown'):.3f} "
            f"DECILE_MONO={agg.get('decile_monotonicity_spearman') if agg.get('decile_monotonicity_spearman') is not None else None} "
            f"IC_DISP_CORR={agg.get('ic_dispersion_corr') if agg.get('ic_dispersion_corr') is not None else None} "
            f"IC_VIX_CORR={agg.get('ic_vix_corr') if agg.get('ic_vix_corr') is not None else None} "
            f"CSV={csv_path}"
        )
        print(f"ROLLING_IC_CSV={rolling_ic_path}")
        print(f"PROBABILITY_BINS_CSV={prob_bins_path}")
        print(f"IC_VIX_DISPERSION_CSV={triplet_path}")
        print(f"REGIME_IC_JSON={regime_ic_path}")
        print(f"VIX_BUCKET_IC_JSON={vix_ic_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
