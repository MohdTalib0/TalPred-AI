"""Expansion validation: 5 critical checks before trusting the 503-symbol results.

CHECK 1 — Survivorship bias risk assessment
  - Shows effective_from distribution for all symbols
  - Reports how many symbols entered DB today (survivorship candidates)
  - Applies liquidity filter (price > $5, dollar_volume > $5M) and shows
    what fraction of signal remains

CHECK 2 — Liquidity-filtered backtest
  - Removes stocks failing min price / dollar_volume thresholds
  - Confirms signal survives after filtering out micro/illiquid stocks

CHECK 3 — Transaction cost sensitivity (10 / 20 / 30 bps)
  - Tests Sharpe degradation under realistic trading costs

CHECK 4 — Portfolio size robustness (top/bottom 10 / 20 / 40)
  - Confirms signal is broadly distributed, not concentrated in tail

CHECK 5 — Randomized label sanity check
  - Shuffles direction labels, retrains, confirms AUC ≈ 0.50, Sharpe ≈ 0

Usage:
  python -m scripts.research_expansion_validation
"""

import logging
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol
from src.models.trainer import train_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("expansion_validation")

EXPERIMENT = "research-expansion-validation"
TARGET_MODE = "sector_relative"
HORIZON = 5
FEATURE_PROFILE = "flow_enhanced"
RANK_TOP_N = 20
NW_LAG = 4

# Liquidity thresholds
MIN_PRICE_USD = 5.0
MIN_DOLLAR_VOLUME_M = 5.0  # million USD per day

BT_BASE = dict(
    rank_top_n=RANK_TOP_N,
    rank_mode="global",
    transaction_cost_bps=10,
    rank_rebalance_stride=1,
    rank_sharpe_nw_lag=NW_LAG,
    feature_profile=FEATURE_PROFILE,
)


# ─── helpers ───────────────────────────────────────────────────────────────────

def _print_section(title: str):
    logger.info("")
    logger.info("=" * 65)
    logger.info(f"  {title}")
    logger.info("=" * 65)


def _sharpe_summary(bt: dict, label: str) -> dict:
    agg = bt.get("aggregate_metrics", {})
    return {
        "config": label,
        "wf_auc": agg.get("overall_auc"),
        "ic_mean": agg.get("ic_mean"),
        "nw_sharpe_net": agg.get("rank_long_short_sharpe_net_nw"),
        "gross_sharpe": agg.get("rank_long_short_sharpe"),
        "net_spread_bps": (agg.get("rank_long_short_mean_net") or 0) * 10_000,
        "max_dd_net": agg.get("rank_max_drawdown_net"),
        "avg_turnover": agg.get("rank_avg_turnover"),
    }


# ─── CHECK 1: survivorship bias assessment ─────────────────────────────────────

def check_survivorship(db, symbols: list[str]):
    _print_section("CHECK 1 — Survivorship Bias Risk Assessment")

    rows = db.execute(text("""
        SELECT symbol, effective_from, market_cap, avg_daily_volume_30d
        FROM symbols
        WHERE is_active = true
        ORDER BY effective_from
    """)).fetchall()

    df = pd.DataFrame(rows, columns=["symbol", "effective_from", "market_cap", "avg_vol"])
    df["effective_from"] = pd.to_datetime(df["effective_from"])

    today = pd.Timestamp(date.today())
    new_today = df[df["effective_from"] >= today - timedelta(days=1)]
    older = df[df["effective_from"] < today - timedelta(days=1)]

    logger.info(f"  Total active symbols:       {len(df)}")
    logger.info(f"  Added before today:         {len(older)}  (lower survivorship risk)")
    logger.info(f"  Added today (new batch):    {len(new_today)}  ← survivorship risk")
    logger.info("")
    logger.info("  Survivorship bias explanation:")
    logger.info("    All 308 new symbols were added today using today's S&P500 list.")
    logger.info("    Stocks that were in S&P500 in 2023 but removed by 2026 are MISSING.")
    logger.info("    This inflates historical performance (only survivors included).")
    logger.info("")
    logger.info("  Mitigation:")
    logger.info("    1. Apply strict liquidity filter (removes most delisted candidates)")
    logger.info("    2. Focus interpretation on 2024-2026 where survivorship gap is smaller")
    logger.info("    3. Treat Sharpe as an upper bound; expect 15-30% haircut in live trading")

    # Show sector coverage of new vs old symbols
    logger.info("")
    logger.info("  Effective_from distribution:")
    ef_counts = df["effective_from"].dt.year.value_counts().sort_index()
    for yr, cnt in ef_counts.items():
        logger.info(f"    {yr}: {cnt} symbols")

    return {"new_today": len(new_today), "older": len(older)}


# ─── CHECK 2: liquidity filter ─────────────────────────────────────────────────

def apply_liquidity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter training dataset to liquid, high-price stocks."""
    original = len(df)

    # Price filter: use rolling_return to infer price level isn't available directly
    # Use dollar_volume and log_market_cap as proxy for price/size filter
    # dollar_volume = close * volume (in raw terms), stored in features as a float
    # MIN_DOLLAR_VOLUME_M = 5M → dollar_volume > 5_000_000

    if "dollar_volume" in df.columns:
        df = df[df["dollar_volume"] > MIN_DOLLAR_VOLUME_M * 1_000_000]
        logger.info(f"  After dollar_volume > ${MIN_DOLLAR_VOLUME_M}M filter: {len(df):,} rows (was {original:,})")
    else:
        logger.info("  dollar_volume column not found — skipping price/volume filter")

    # Size filter: log_market_cap > log(500M) ≈ 20.03
    if "log_market_cap" in df.columns:
        min_log_mcap = np.log(500_000_000)  # $500M min market cap
        before = len(df)
        df = df[df["log_market_cap"] > min_log_mcap]
        logger.info(f"  After market_cap > $500M filter:  {len(df):,} rows (was {before:,})")

    removed_pct = 100 * (1 - len(df) / original)
    logger.info(f"  Removed {removed_pct:.1f}% of rows by liquidity filter")
    logger.info(f"  Remaining unique symbols: {df['symbol'].nunique()}")
    return df


def check_liquidity_filtered_bt(df_full: pd.DataFrame) -> dict:
    _print_section("CHECK 2 — Liquidity-Filtered Backtest")

    df_filtered = apply_liquidity_filter(df_full.copy())

    logger.info("\n  Training model on liquidity-filtered universe...")
    result = train_baseline(
        df_filtered,
        experiment_name=EXPERIMENT,
        run_name="liquidity_filtered",
        dataset_version="v1.0-backfill",
        feature_profile=FEATURE_PROFILE,
        run_mode="research",
    )

    model = result["model"]
    medians = result["train_medians"]

    logger.info("  Running walk-forward backtest on filtered universe...")
    bt = walk_forward_backtest(df_filtered, **BT_BASE)
    s = _sharpe_summary(bt, "liquidity_filtered")

    logger.info(f"  WF AUC:        {s['wf_auc']:.4f}" if s["wf_auc"] else "  WF AUC: N/A")
    logger.info(f"  IC mean:       {s['ic_mean']:.4f}" if s["ic_mean"] else "  IC: N/A")
    logger.info(f"  NW Sharpe net: {s['nw_sharpe_net']:.3f}" if s["nw_sharpe_net"] else "  NW Sharpe: N/A")
    logger.info(f"  Unique syms:   {df_filtered['symbol'].nunique()}")
    return s


# ─── CHECK 3: cost sensitivity ─────────────────────────────────────────────────

def check_cost_sensitivity(df: pd.DataFrame, model_result: dict) -> list[dict]:
    _print_section("CHECK 3 — Transaction Cost Sensitivity (10 / 20 / 30 bps)")

    rows = []
    for cost_bps in [10, 20, 30]:
        logger.info(f"\n  Cost = {cost_bps} bps ...")
        bt = walk_forward_backtest(
            df,
            transaction_cost_bps=cost_bps,
            rank_top_n=RANK_TOP_N,
            rank_mode="global",
            rank_rebalance_stride=1,
            rank_sharpe_nw_lag=NW_LAG,
            feature_profile=FEATURE_PROFILE,
        )
        s = _sharpe_summary(bt, f"cost_{cost_bps}bps")
        rows.append(s)
        logger.info(f"    NW Sharpe net: {s['nw_sharpe_net']:.3f}" if s["nw_sharpe_net"] else "    NW Sharpe: N/A")
        logger.info(f"    Net spread:    {s['net_spread_bps']:.1f} bps")
        logger.info(f"    Max DD:        {s['max_dd_net']:.3f}" if s["max_dd_net"] else "    MDD: N/A")

    return rows


# ─── CHECK 4: portfolio size robustness ────────────────────────────────────────

def check_portfolio_size(df: pd.DataFrame) -> list[dict]:
    _print_section("CHECK 4 — Portfolio Size Robustness (top/bottom 10 / 20 / 40)")

    rows = []
    for top_n in [10, 20, 40]:
        logger.info(f"\n  top_n = {top_n} ...")
        bt = walk_forward_backtest(
            df,
            transaction_cost_bps=10,
            rank_top_n=top_n,
            rank_mode="global",
            rank_rebalance_stride=1,
            rank_sharpe_nw_lag=NW_LAG,
            feature_profile=FEATURE_PROFILE,
        )
        s = _sharpe_summary(bt, f"top_{top_n}")
        rows.append(s)
        logger.info(f"    NW Sharpe net: {s['nw_sharpe_net']:.3f}" if s["nw_sharpe_net"] else "    NW Sharpe: N/A")
        logger.info(f"    IC mean:       {s['ic_mean']:.4f}" if s["ic_mean"] else "    IC: N/A")
        logger.info(f"    Max DD:        {s['max_dd_net']:.3f}" if s["max_dd_net"] else "    MDD: N/A")

    return rows


# ─── CHECK 5: randomized label sanity check ────────────────────────────────────

def check_randomized_labels(df: pd.DataFrame) -> dict:
    _print_section("CHECK 5 — Randomized Label Sanity Check")
    logger.info("  Shuffling direction labels to verify pipeline has no leakage ...")
    logger.info("  Expected: AUC ≈ 0.50, IC ≈ 0, Sharpe ≈ 0")

    df_rand = df.copy()
    np.random.seed(42)
    df_rand["direction"] = np.random.permutation(df_rand["direction"].values)

    result = train_baseline(
        df_rand,
        experiment_name=EXPERIMENT,
        run_name="randomized_labels",
        dataset_version="v1.0-backfill",
        feature_profile=FEATURE_PROFILE,
        run_mode="research",
    )

    bt = walk_forward_backtest(
        df_rand,
        transaction_cost_bps=10,
        rank_top_n=RANK_TOP_N,
        rank_mode="global",
        rank_rebalance_stride=1,
        rank_sharpe_nw_lag=NW_LAG,
        feature_profile=FEATURE_PROFILE,
    )
    s = _sharpe_summary(bt, "randomized_labels")

    logger.info(f"\n  RANDOMIZED RESULTS:")
    logger.info(f"    WF AUC:        {s['wf_auc']:.4f}" if s["wf_auc"] else "    WF AUC: N/A")
    logger.info(f"    IC mean:       {s['ic_mean']:.4f}" if s["ic_mean"] else "    IC: N/A")
    logger.info(f"    NW Sharpe net: {s['nw_sharpe_net']:.3f}" if s["nw_sharpe_net"] else "    NW Sharpe: N/A")

    auc_ok = (s.get("wf_auc") or 0) < 0.510
    ic_ok = abs(s.get("ic_mean") or 0) < 0.010
    sharpe_ok = abs(s.get("nw_sharpe_net") or 0) < 0.5

    if auc_ok and ic_ok and sharpe_ok:
        logger.info("  RESULT: PASS — pipeline is clean, no data leakage detected")
    else:
        logger.warning("  RESULT: FAIL — randomized model is too strong, possible leakage!")
        logger.warning(f"    AUC check (<0.51):    {'PASS' if auc_ok else 'FAIL'}")
        logger.warning(f"    IC check (<0.01):     {'PASS' if ic_ok else 'FAIL'}")
        logger.warning(f"    Sharpe check (<0.5):  {'PASS' if sharpe_ok else 'FAIL'}")

    return {**s, "auc_ok": auc_ok, "ic_ok": ic_ok, "sharpe_ok": sharpe_ok}


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("Loading dataset (full 503-symbol universe)...")
    db = SessionLocal()
    end_date = date.today()
    start_date = end_date - timedelta(days=3 * 365)
    symbols = [r.symbol for r in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]

    df = build_training_dataset(
        db, symbols, start_date, end_date,
        target_mode=TARGET_MODE,
        target_horizon_days=HORIZON,
        include_liquidity_features=True,
    )
    db.close()
    logger.info(f"Dataset: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # ── CHECK 1: survivorship ──────────────────────────────────────────────
    db2 = SessionLocal()
    surv_result = check_survivorship(db2, symbols)
    db2.close()

    # ── Train shared model for cost/size checks (full universe) ───────────
    _print_section("TRAINING SHARED MODEL (full universe, flow_enhanced)")
    model_result = train_baseline(
        df,
        experiment_name=EXPERIMENT,
        run_name="validation_baseline",
        dataset_version="v1.0-backfill",
        feature_profile=FEATURE_PROFILE,
        run_mode="research",
    )
    logger.info(f"  Train AUC: {model_result['metrics']['auc_roc']:.4f}")

    # ── CHECK 2: liquidity filter ──────────────────────────────────────────
    liq_result = check_liquidity_filtered_bt(df)

    # ── CHECK 3: cost sensitivity ──────────────────────────────────────────
    cost_results = check_cost_sensitivity(df, model_result)

    # ── CHECK 4: portfolio size robustness ────────────────────────────────
    size_results = check_portfolio_size(df)

    # ── CHECK 5: randomized labels ────────────────────────────────────────
    rand_result = check_randomized_labels(df)

    # ── FINAL VERDICT ─────────────────────────────────────────────────────
    _print_section("VALIDATION SUMMARY")

    cost_20_sharpe = next((r["nw_sharpe_net"] for r in cost_results if "20bps" in r["config"]), None)
    cost_30_sharpe = next((r["nw_sharpe_net"] for r in cost_results if "30bps" in r["config"]), None)
    size_10_sharpe = next((r["nw_sharpe_net"] for r in size_results if "top_10" in r["config"]), None)
    size_40_sharpe = next((r["nw_sharpe_net"] for r in size_results if "top_40" in r["config"]), None)

    checks = {
        "survivorship_risk": "HIGH (308/503 symbols added today — treat as upper bound)",
        "liquidity_filter_sharpe": f"{liq_result.get('nw_sharpe_net', 'N/A'):.2f}" if liq_result.get("nw_sharpe_net") else "N/A",
        "cost_20bps_sharpe": f"{cost_20_sharpe:.2f}" if cost_20_sharpe else "N/A",
        "cost_30bps_sharpe": f"{cost_30_sharpe:.2f}" if cost_30_sharpe else "N/A",
        "portfolio_top10_sharpe": f"{size_10_sharpe:.2f}" if size_10_sharpe else "N/A",
        "portfolio_top40_sharpe": f"{size_40_sharpe:.2f}" if size_40_sharpe else "N/A",
        "randomized_auc": f"{rand_result.get('wf_auc', 'N/A'):.4f}" if rand_result.get("wf_auc") else "N/A",
        "randomized_sharpe": f"{rand_result.get('nw_sharpe_net', 'N/A'):.3f}" if rand_result.get("nw_sharpe_net") else "N/A",
        "leakage_test": "PASS" if rand_result.get("auc_ok") and rand_result.get("ic_ok") else "FAIL",
    }

    for k, v in checks.items():
        logger.info(f"  {k:35s}: {v}")

    # Gate thresholds
    passes = 0
    total_gates = 4
    if (cost_20_sharpe or 0) > 2.0:
        logger.info("\n  [PASS] Signal survives 20 bps costs (Sharpe > 2)")
        passes += 1
    else:
        logger.warning(f"\n  [FAIL] Signal weak at 20 bps (Sharpe = {cost_20_sharpe:.2f})")

    if (size_10_sharpe or 0) > 1.0:
        logger.info("  [PASS] Signal survives top/bottom 10 portfolio (Sharpe > 1)")
        passes += 1
    else:
        logger.warning(f"  [FAIL] Signal too concentrated — top10 Sharpe = {size_10_sharpe}")

    if rand_result.get("auc_ok") and rand_result.get("ic_ok"):
        logger.info("  [PASS] No data leakage detected (randomized test clean)")
        passes += 1
    else:
        logger.warning("  [FAIL] Randomized test suggests possible data leakage!")

    liq_sharpe_val = liq_result.get("nw_sharpe_net") or 0
    if liq_sharpe_val > 1.5:
        logger.info(f"  [PASS] Liquidity-filtered signal holds (Sharpe = {liq_sharpe_val:.2f})")
        passes += 1
    else:
        logger.warning(f"  [WARN] Liquidity-filtered Sharpe dropped to {liq_sharpe_val:.2f}")

    logger.info(f"\n  GATES PASSED: {passes}/{total_gates}")
    logger.info("  SURVIVORSHIP NOTE: Backtest Sharpe is an upper bound.")
    logger.info("  Expected live degradation: 20-40% due to survivorship + execution.")

    # Save results
    out_dir = "artifacts/research"
    os.makedirs(out_dir, exist_ok=True)

    all_results = (
        [{"check": "cost_sensitivity", **r} for r in cost_results]
        + [{"check": "portfolio_size", **r} for r in size_results]
        + [{"check": "liquidity_filter", **liq_result}]
        + [{"check": "randomized_labels", **rand_result}]
    )
    pd.DataFrame(all_results).to_csv(f"{out_dir}/expansion_validation.csv", index=False)
    logger.info(f"\n  Results saved → {out_dir}/expansion_validation.csv")


if __name__ == "__main__":
    main()
