"""Production diagnostics: capacity test, crash simulation, feature stability.

Run:  python -m scripts.prod_diagnostics
"""
import sys
import logging
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol
from src.models.trainer import DEFAULT_PARAMS, prepare_features, FEATURE_PROFILES


# ───────────────────────────────────────────────────────────
# 1. CAPACITY TEST
# ───────────────────────────────────────────────────────────
def capacity_test(df: pd.DataFrame) -> dict:
    """Simulate at different capital levels with realistic slippage.

    For each capital level, compute the participation rate per stock
    (position_value / ADV), estimate market-impact slippage, and
    report the net Sharpe after costs.
    """
    print("\n" + "=" * 60)
    print("1. CAPACITY TEST")
    print("=" * 60)

    capital_levels = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]
    results = {}

    top_n = 20
    cost_bps_levels = {
        100_000: 5.0,
        500_000: 8.0,
        1_000_000: 10.0,
        5_000_000: 20.0,
        10_000_000: 35.0,
        50_000_000: 60.0,
    }

    for capital in capital_levels:
        cost_bps = cost_bps_levels[capital]

        bt = walk_forward_backtest(
            df,
            min_train_days=252,
            step_days=21,
            rank_top_n=top_n,
            transaction_cost_bps=cost_bps,
            rank_rebalance_stride=5,
            feature_profile="cross_sectional_alpha",
        )

        if "error" in bt:
            results[capital] = {"error": bt["error"]}
            continue

        a = bt["aggregate_metrics"]
        sharpe = a.get("rank_long_short_sharpe_net", 0)
        mdd = a.get("rank_max_drawdown_net", 0)
        turnover = a.get("rank_avg_turnover", 0)

        per_position = capital / (top_n * 2)
        adv_median = 5_000_000
        participation = per_position / adv_median * 100

        results[capital] = {
            "sharpe": sharpe,
            "mdd": mdd,
            "turnover": turnover,
            "cost_bps": cost_bps,
            "participation_pct": participation,
        }
        print(
            f"  ${capital:>12,}  |  cost={cost_bps:5.1f}bps  "
            f"|  Sharpe={sharpe:+.2f}  |  MDD={mdd:+.2%}  "
            f"|  participation={participation:.2f}%"
        )

    sharpe_at_1m = results.get(1_000_000, {}).get("sharpe", 0)
    sharpe_at_10m = results.get(10_000_000, {}).get("sharpe", 0)
    sharpe_at_50m = results.get(50_000_000, {}).get("sharpe", 0)

    print("\n  Summary:")
    if sharpe_at_10m > 1.0:
        print("  Strategy holds up at $10M capacity.")
    elif sharpe_at_1m > 1.0:
        print("  Strategy is viable at $1M-$5M. Degrades beyond that.")
    else:
        print("  Small capacity strategy: viable at <$1M only.")

    if sharpe_at_50m > 0.5:
        print("  Even $50M shows positive risk-adjusted returns.")
    elif sharpe_at_50m <= 0:
        print("  $50M kills the alpha — this is a small-cap strategy.")

    return results


# ───────────────────────────────────────────────────────────
# 2. CRASH SIMULATION
# ───────────────────────────────────────────────────────────
def crash_simulation(df: pd.DataFrame) -> dict:
    """Test strategy during worst-case market regimes.

    Identifies high-VIX, high-drawdown periods and reports model IC
    and L/S spread during those windows.
    """
    print("\n" + "=" * 60)
    print("2. CRASH / STRESS SIMULATION")
    print("=" * 60)

    bt = walk_forward_backtest(
        df,
        min_train_days=252,
        step_days=21,
        rank_top_n=20,
        transaction_cost_bps=10.0,
        rank_rebalance_stride=5,
        feature_profile="cross_sectional_alpha",
    )

    if "error" in bt:
        print(f"  ERROR: {bt['error']}")
        return {}

    preds = bt["predictions"]
    a = bt["aggregate_metrics"]

    if "vix_level" not in preds.columns or "target_value" not in preds.columns:
        print("  Cannot run crash sim — missing vix_level or target_value")
        return {}

    results = {}

    # Regime breakdown
    vix_bins = {
        "low_vol (VIX<18)": preds[preds["vix_level"] < 18],
        "normal (18<=VIX<25)": preds[(preds["vix_level"] >= 18) & (preds["vix_level"] < 25)],
        "elevated (25<=VIX<30)": preds[(preds["vix_level"] >= 25) & (preds["vix_level"] < 30)],
        "crisis (VIX>=30)": preds[preds["vix_level"] >= 30],
    }

    print("\n  Regime   | Days | IC     | L/S Spread | Accuracy")
    print("  " + "-" * 58)

    for label, sub in vix_bins.items():
        if sub.empty or len(sub) < 50:
            print(f"  {label:30s} |  N/A | N/A    | N/A        | N/A")
            continue

        daily_ic = []
        daily_ls = []
        for _, day_df in sub.groupby("date"):
            valid = day_df.dropna(subset=["target_value"])
            if len(valid) < 20:
                continue
            ic = float(valid["probability_up"].corr(valid["target_value"], method="spearman"))
            daily_ic.append(ic)
            sorted_day = valid.sort_values("probability_up", ascending=False)
            long_ret = sorted_day.head(20)["target_value"].mean()
            short_ret = sorted_day.tail(20)["target_value"].mean()
            daily_ls.append(float(long_ret - short_ret))

        if not daily_ic:
            continue

        ic_mean = np.mean(daily_ic)
        ls_mean = np.mean(daily_ls)
        acc = accuracy_score_safe(sub)

        results[label] = {
            "days": len(daily_ic),
            "ic_mean": ic_mean,
            "ls_spread_bps": ls_mean * 10000,
            "accuracy": acc,
        }
        print(
            f"  {label:30s} | {len(daily_ic):4d} | {ic_mean:+.4f} "
            f"| {ls_mean * 10000:+8.1f}bps | {acc:.3f}"
        )

    # Worst drawdown periods
    print("\n  --- Drawdown Analysis ---")
    daily_pnl = []
    for dt, day_df in preds.groupby("date"):
        valid = day_df.dropna(subset=["target_value"]).sort_values("probability_up", ascending=False)
        if len(valid) < 40:
            continue
        long_ret = valid.head(20)["target_value"].mean()
        short_ret = valid.tail(20)["target_value"].mean()
        daily_pnl.append({"date": dt, "pnl": float(long_ret - short_ret)})

    if daily_pnl:
        pnl_df = pd.DataFrame(daily_pnl).sort_values("date")
        cumulative = (1 + pnl_df["pnl"]).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative / rolling_max) - 1
        pnl_df["drawdown"] = drawdown.values
        pnl_df["cumulative"] = cumulative.values

        worst_dd = drawdown.min()
        worst_dd_date = pnl_df.iloc[drawdown.idxmin()]["date"]
        print(f"  Max drawdown: {worst_dd:.2%} (around {worst_dd_date})")

        worst_5 = pnl_df.nsmallest(5, "pnl")
        print(f"  5 worst days:")
        for _, r in worst_5.iterrows():
            print(f"    {r['date']}  PnL: {r['pnl']:+.4f}  DD: {r['drawdown']:.2%}")

        recovery = None
        dd_idx = drawdown.idxmin()
        post_dd = cumulative.iloc[dd_idx:]
        peak = rolling_max.iloc[dd_idx]
        recovered = post_dd[post_dd >= peak]
        if len(recovered) > 0:
            recovery_days = recovered.index[0] - dd_idx
            recovery = recovery_days
        print(f"  Recovery from max DD: {recovery if recovery else 'Not recovered'} periods")

    results["max_drawdown"] = float(worst_dd) if daily_pnl else None
    return results


def accuracy_score_safe(df):
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(df["actual"], df["predicted"]))


# ───────────────────────────────────────────────────────────
# 3. FEATURE STABILITY
# ───────────────────────────────────────────────────────────
def feature_stability(df: pd.DataFrame) -> dict:
    """Check if feature importances are stable across walk-forward folds.

    Trains a model at each fold and records feature importance rankings.
    Stable features = consistently important across time.
    Fragile features = importance varies wildly.
    """
    print("\n" + "=" * 60)
    print("3. FEATURE IMPORTANCE STABILITY")
    print("=" * 60)

    profile = "cross_sectional_alpha"
    feature_cols = FEATURE_PROFILES[profile]
    params = DEFAULT_PARAMS.copy()

    df = df.sort_values("target_session_date").reset_index(drop=True)
    dates = sorted(df["target_session_date"].unique())
    min_train = 252
    step = 21

    fold_importances = []
    fold_labels = []

    idx = min_train
    while idx < len(dates):
        end_idx = min(idx + step, len(dates))
        train_end = dates[idx - 1]
        train_df = df[df["target_session_date"] <= train_end]

        if len(train_df) < 100:
            idx += step
            continue

        X, y, _ = prepare_features(train_df, feature_profile=profile)
        model = xgb.XGBClassifier(**params)
        model.fit(X, y, verbose=False)

        importance = model.feature_importances_
        cols_used = list(X.columns)

        imp_dict = dict(zip(cols_used, importance))
        fold_importances.append(imp_dict)
        fold_labels.append(str(train_end)[:10])

        idx += step

    if not fold_importances:
        print("  No folds computed!")
        return {}

    all_features = sorted(set().union(*[d.keys() for d in fold_importances]))
    imp_matrix = pd.DataFrame(
        [{f: d.get(f, 0.0) for f in all_features} for d in fold_importances],
        index=fold_labels,
    )

    rank_matrix = imp_matrix.rank(axis=1, ascending=False)

    print(f"\n  Folds analyzed: {len(fold_importances)}")
    print(f"  Features tracked: {len(all_features)}")

    mean_imp = imp_matrix.mean()
    std_imp = imp_matrix.std()
    cv_imp = (std_imp / mean_imp.clip(lower=1e-8)).clip(upper=10)

    mean_rank = rank_matrix.mean()
    std_rank = rank_matrix.std()

    summary = pd.DataFrame({
        "mean_importance": mean_imp,
        "std_importance": std_imp,
        "cv": cv_imp,
        "mean_rank": mean_rank,
        "std_rank": std_rank,
    }).sort_values("mean_importance", ascending=False)

    print("\n  Top 15 features by mean importance:")
    print(f"  {'Feature':40s} | {'MeanImp':>8s} | {'CV':>5s} | {'MeanRank':>8s} | {'RankStd':>7s} | Stable?")
    print("  " + "-" * 85)

    stable_count = 0
    for feat, row in summary.head(15).iterrows():
        stable = "YES" if row["cv"] < 0.5 and row["std_rank"] < 5 else " no"
        if stable == "YES":
            stable_count += 1
        print(
            f"  {feat:40s} | {row['mean_importance']:8.4f} | {row['cv']:5.2f} "
            f"| {row['mean_rank']:8.1f} | {row['std_rank']:7.1f} | {stable}"
        )

    print(f"\n  Stable features (of top 15): {stable_count}/15")

    fragile = summary[(summary["cv"] > 1.0) & (summary["mean_importance"] > 0.01)]
    if len(fragile) > 0:
        print(f"\n  WARNING: {len(fragile)} fragile features (high importance but unstable):")
        for feat, row in fragile.iterrows():
            print(f"    {feat}: CV={row['cv']:.2f}, rank_std={row['std_rank']:.1f}")
    else:
        print("\n  No fragile features detected.")

    # First fold vs last fold — drift check
    first_top5 = set(rank_matrix.iloc[0].nsmallest(5).index)
    last_top5 = set(rank_matrix.iloc[-1].nsmallest(5).index)
    overlap = first_top5 & last_top5
    print(f"\n  First fold top-5: {sorted(first_top5)}")
    print(f"  Last fold top-5:  {sorted(last_top5)}")
    print(f"  Overlap: {len(overlap)}/5 ({', '.join(sorted(overlap)) if overlap else 'NONE'})")

    if len(overlap) >= 3:
        print("  Feature importance is STABLE across time.")
    elif len(overlap) >= 1:
        print("  Moderate drift — some features rotated in/out.")
    else:
        print("  WARNING: Complete feature rotation — model may be fragile!")

    return {
        "folds": len(fold_importances),
        "stable_top15": stable_count,
        "drift_overlap": len(overlap),
        "top_features": list(summary.head(10).index),
    }


# ───────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────
def main():
    db = SessionLocal()
    symbols = [
        r.symbol
        for r in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()
    ]
    end_date = date.today()
    start_date = end_date - timedelta(days=3 * 365)

    print("Building dataset (market_relative, 5d horizon)...")
    df = build_training_dataset(
        db,
        symbols,
        start_date,
        end_date,
        target_mode="market_relative",
        include_liquidity_features=True,
        target_horizon_days=5,
    )
    print(f"Dataset: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    cap_results = capacity_test(df)
    crash_results = crash_simulation(df)
    stability_results = feature_stability(df)

    print("\n" + "=" * 60)
    print("ALL DIAGNOSTICS COMPLETE")
    print("=" * 60)

    db.close()


if __name__ == "__main__":
    main()
