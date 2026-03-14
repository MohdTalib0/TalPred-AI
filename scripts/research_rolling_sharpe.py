"""3-month rolling Sharpe analysis for the flow_enhanced model.

Answers the promotion readiness question:
  - How long did the 2024 weak period last?
  - How quickly did the signal recover?
  - Are loss periods short enough that the IC deployment guard is sufficient?

Computes:
  - 63-day (≈ 3-month) rolling annualized Sharpe on net long/short returns
  - Max consecutive days below zero rolling Sharpe (worst drought)
  - Recovery time: days from trough to next positive Sharpe
  - Rolling Sharpe chart saved as PNG

Usage:
  python -m scripts.research_rolling_sharpe
"""

import json
import logging
import os
from datetime import date, timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

matplotlib.use("Agg")
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
logger = logging.getLogger("research_rolling_sharpe")

WINDOW = 63   # ~3 calendar months of trading days
ANNUALIZATION = np.sqrt(252)
OUT_DIR = "artifacts/research"


def _rolling_sharpe(returns: pd.Series, window: int = WINDOW) -> pd.Series:
    """Annualized Sharpe on a rolling window."""
    roll_mean = returns.rolling(window, min_periods=window // 2).mean()
    roll_std = returns.rolling(window, min_periods=window // 2).std(ddof=1)
    return (roll_mean / roll_std.replace(0, np.nan)) * ANNUALIZATION


def _drought_stats(roll_sharpe: pd.Series) -> dict:
    """Compute drought (consecutive negative-Sharpe) and recovery statistics."""
    is_neg = (roll_sharpe < 0).astype(int)

    # Consecutive negative stretches
    streaks = []
    count = 0
    for v in is_neg:
        if v == 1:
            count += 1
        else:
            if count > 0:
                streaks.append(count)
            count = 0
    if count > 0:
        streaks.append(count)

    # Recovery time: from last negative to next positive
    recovery_times = []
    in_drought = False
    drought_start = None
    for i, (dt, v) in enumerate(roll_sharpe.items()):
        if not in_drought and v < 0:
            in_drought = True
            drought_start = i
        elif in_drought and (v >= 0 or np.isnan(v)):
            if drought_start is not None:
                recovery_times.append(i - drought_start)
            in_drought = False
            drought_start = None

    return {
        "max_consecutive_negative_days": int(max(streaks)) if streaks else 0,
        "avg_drought_length": float(np.mean(streaks)) if streaks else 0,
        "n_droughts": len(streaks),
        "avg_recovery_days": float(np.mean(recovery_times)) if recovery_times else 0,
        "max_recovery_days": int(max(recovery_times)) if recovery_times else 0,
    }


def _plot_rolling_sharpe(roll_gross: pd.Series, roll_net: pd.Series, out_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        "flow_enhanced — 3-Month Rolling Sharpe (63-day window)\n"
        "sector_relative target | horizon=5d | top/bottom 20 | 10 bps cost",
        fontsize=12, fontweight="bold",
    )

    for ax, series, label, color in [
        (axes[0], roll_gross, "Gross L/S Sharpe", "#2196F3"),
        (axes[1], roll_net, "Net L/S Sharpe (after 10 bps cost)", "#4CAF50"),
    ]:
        ax.plot(series.index, series.values, color=color, linewidth=1.4, label=label)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(1.0, color="orange", linewidth=0.8, linestyle=":", alpha=0.7, label="Sharpe = 1.0")
        ax.axhline(2.0, color="red", linewidth=0.8, linestyle=":", alpha=0.5, label="Sharpe = 2.0")

        # Shade positive / negative regions
        ax.fill_between(series.index, series.values, 0,
                         where=(series.values >= 0), alpha=0.15, color="green", label="Positive")
        ax.fill_between(series.index, series.values, 0,
                         where=(series.values < 0), alpha=0.15, color="red", label="Negative")

        # Annotate 2024 region
        idx_2024 = series.index[(series.index >= "2024-01-01") & (series.index < "2025-01-01")]
        if len(idx_2024):
            ax.axvspan(idx_2024[0], idx_2024[-1], alpha=0.05, color="gray", label="2024")
            ax.text(idx_2024[0], ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else -1.5,
                    "2024", fontsize=8, color="gray", alpha=0.8)

        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Date")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {out_path}")


def main():
    logger.info("Loading dataset...")
    db = SessionLocal()
    end_date = date.today()
    start_date = end_date - timedelta(days=3 * 365)
    symbols = [row.symbol for row in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]
    df = build_training_dataset(
        db, symbols, start_date, end_date,
        target_mode="sector_relative",
        target_horizon_days=5,
        include_liquidity_features=True,
    )
    db.close()
    logger.info(f"Dataset: {len(df):,} rows, {df['target_session_date'].nunique()} dates")

    logger.info("Training flow_enhanced model...")
    train_baseline(
        df,
        experiment_name="research-rolling-sharpe",
        run_name="rolling_sharpe_flow_enhanced",
        dataset_version="v1.0-backfill",
        feature_profile="flow_enhanced",
        run_mode="research",
    )

    logger.info("Running walk-forward backtest...")
    bt = walk_forward_backtest(
        df,
        feature_profile="flow_enhanced",
        rank_top_n=20,
        rank_mode="global",
        transaction_cost_bps=10,
        rank_rebalance_stride=1,
        rank_sharpe_nw_lag=4,
    )

    agg = bt.get("aggregate_metrics", {})
    daily_series = agg.get("rank_daily_series", [])
    if not daily_series:
        logger.error("No daily series returned from backtest. Check walk_forward_backtest.")
        return

    # Build a clean daily DataFrame
    ret_df = pd.DataFrame(daily_series)
    ret_df["date"] = pd.to_datetime(ret_df["date"])
    ret_df = ret_df.sort_values("date").set_index("date")
    ret_df = ret_df.asfreq("B").fillna(0)  # business-day frequency, fill non-trading days with 0

    logger.info(f"Daily return series: {len(ret_df)} days from {ret_df.index[0].date()} to {ret_df.index[-1].date()}")

    # Rolling Sharpe
    roll_gross = _rolling_sharpe(ret_df["long_short"])
    roll_net = _rolling_sharpe(ret_df["long_short_net"])

    # Drought/recovery stats
    drought_gross = _drought_stats(roll_gross.dropna())
    drought_net = _drought_stats(roll_net.dropna())

    # Print key findings
    logger.info("\n" + "=" * 70)
    logger.info("3-MONTH ROLLING SHARPE — ROBUSTNESS REPORT")
    logger.info("=" * 70)

    logger.info(f"\nFull-period stats:")
    logger.info(f"  Gross rolling Sharpe — mean: {roll_gross.mean():.3f}  min: {roll_gross.min():.3f}  max: {roll_gross.max():.3f}")
    logger.info(f"  Net   rolling Sharpe — mean: {roll_net.mean():.3f}  min: {roll_net.min():.3f}  max: {roll_net.max():.3f}")

    # Year-by-year rolling Sharpe snapshot (end-of-year value)
    logger.info(f"\nYear-end rolling Sharpe (net):")
    for yr in ["2024", "2025", "2026"]:
        yr_vals = roll_net[roll_net.index.year == int(yr)]
        if yr_vals.empty:
            continue
        q25 = yr_vals.quantile(0.25)
        median = yr_vals.median()
        q75 = yr_vals.quantile(0.75)
        pct_positive = (yr_vals > 0).mean() * 100
        logger.info(f"  {yr}: median={median:.2f}  Q25={q25:.2f}  Q75={q75:.2f}  positive_windows={pct_positive:.0f}%")

    logger.info(f"\nDrought analysis (net, window={WINDOW}d):")
    logger.info(f"  Max consecutive negative-Sharpe days: {drought_net['max_consecutive_negative_days']}")
    logger.info(f"  # of distinct droughts:              {drought_net['n_droughts']}")
    logger.info(f"  Avg drought length (days):            {drought_net['avg_drought_length']:.0f}")
    logger.info(f"  Avg recovery time (days):             {drought_net['avg_recovery_days']:.0f}")
    logger.info(f"  Max recovery time (days):             {drought_net['max_recovery_days']}")

    # Promotion signal
    logger.info("\nPROMOTION ASSESSMENT:")
    max_drought = drought_net["max_consecutive_negative_days"]
    avg_recovery = drought_net["avg_recovery_days"]
    if max_drought <= 90 and avg_recovery <= 45:
        logger.info("  RESULT: Drought periods are short enough. Regime guard is sufficient.")
        logger.info("  VERDICT: Model is SAFE to promote with --allow-promote + IC guard active.")
    elif max_drought <= 130:
        logger.info(f"  RESULT: Max drought {max_drought}d is moderate. Monitor IC guard carefully.")
        logger.info("  VERDICT: Promote with reduced initial exposure (50%) until IC recovers.")
    else:
        logger.info(f"  RESULT: Max drought {max_drought}d is long. Signal may be genuinely regime-dependent.")
        logger.info("  VERDICT: Hold as candidate. Extend training data before promotion.")

    # Outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = f"{OUT_DIR}/rolling_sharpe_flow_enhanced.csv"
    roll_df = pd.DataFrame({
        "date": roll_net.index,
        "rolling_sharpe_net_63d": roll_net.values,
        "rolling_sharpe_gross_63d": roll_gross.values,
    })
    roll_df.to_csv(csv_path, index=False)
    logger.info(f"\nCSV saved: {csv_path}")

    png_path = f"{OUT_DIR}/rolling_sharpe_flow_enhanced.png"
    _plot_rolling_sharpe(roll_gross, roll_net, png_path)

    results = {
        "window_days": WINDOW,
        "full_period": {
            "mean_net": float(roll_net.mean()),
            "min_net": float(roll_net.min()),
            "max_net": float(roll_net.max()),
            "pct_positive": float((roll_net > 0).mean() * 100),
        },
        "drought_net": drought_net,
        "drought_gross": drought_gross,
        "yearly_median_net": {
            yr: float(roll_net[roll_net.index.year == int(yr)].median())
            for yr in ["2024", "2025", "2026"]
            if not roll_net[roll_net.index.year == int(yr)].empty
        },
    }
    json_path = f"{OUT_DIR}/rolling_sharpe_stats.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Stats saved: {json_path}")


if __name__ == "__main__":
    main()
