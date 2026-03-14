"""Promotion readiness checks for flow_enhanced model.

Runs two targeted robustness checks required before promoting:

  Check 1 — Time Stability
    Yearly Sharpe breakdown: 2023 / 2024 / 2025 / 2026
    Confirms signal is not regime-concentrated.

  Check 2 — Sector Contribution
    Per-sector Sharpe: confirms alpha is broad-based, not one-sector artifact.
    Also flags sector concentration ratio.

  Check 3 — Residual Flow Ablation
    Compares flow_enhanced vs flow_residual profiles.
    Tests whether orthogonalized flow (pure order-flow after removing size/sector)
    is more predictive than raw flow.

Usage:
  python -m scripts.research_promotion_readiness
"""

import json
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
from src.models.trainer import FEATURE_PROFILES, train_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("research_promotion_readiness")

DATASET_KW = dict(
    target_mode="sector_relative",
    target_horizon_days=5,
    include_liquidity_features=True,
)
BT_KW = dict(
    rank_top_n=20,
    rank_mode="global",
    transaction_cost_bps=10,
    rank_rebalance_stride=1,
    rank_sharpe_nw_lag=4,
)

PROMOTION_GATES = {
    "min_nw_sharpe_net": 1.5,
    "min_ic": 0.035,
    "min_wf_auc": 0.52,
    "max_mdd_net": -0.40,
    "max_sector_concentration": 2.5,
    "min_yearly_sharpe": -0.5,
    "min_years_positive": 2,
}


def _print_section(title: str) -> None:
    logger.info(f"\n{'='*70}")
    logger.info(f"  {title}")
    logger.info(f"{'='*70}")


def _check_time_stability(agg: dict) -> dict:
    _print_section("CHECK 1: TIME STABILITY (Yearly Sharpe)")
    yearly = agg.get("rank_yearly_sharpe_net", {})
    if not yearly:
        logger.warning("  No yearly Sharpe data available.")
        return {"passed": False, "yearly": {}}

    logger.info(f"  {'Year':<8} {'Net Sharpe':>12}")
    logger.info(f"  {'-'*22}")
    positive_years = 0
    min_sharpe = float("inf")
    for yr in sorted(yearly.keys()):
        v = yearly[yr]
        flag = "OK" if (v or 0) > 0 else "WEAK"
        logger.info(f"  {yr:<8} {(v or 0):>12.3f}  {flag}")
        if (v or 0) > 0:
            positive_years += 1
        if v is not None:
            min_sharpe = min(min_sharpe, v)

    passed = (
        positive_years >= PROMOTION_GATES["min_years_positive"]
        and min_sharpe >= PROMOTION_GATES["min_yearly_sharpe"]
    )
    verdict = "PASS" if passed else "FAIL"
    logger.info(f"\n  Positive years: {positive_years}/{len(yearly)}  |  Min yearly Sharpe: {min_sharpe:.3f}")
    logger.info(f"  Gate result: {verdict}")
    return {"passed": passed, "yearly": yearly, "positive_years": positive_years, "min_yearly_sharpe": min_sharpe}


def _check_sector_contribution(agg: dict) -> dict:
    _print_section("CHECK 2: SECTOR CONTRIBUTION")
    sharpes = agg.get("sector_sharpes", {})
    spreads = agg.get("sector_spreads_bps", {})
    ranked = agg.get("sector_sharpe_ranked", [])
    concentration = agg.get("sector_concentration_ratio")

    if not sharpes:
        logger.warning("  No sector data. Ensure 'sector' column is in predictions.")
        return {"passed": False}

    logger.info(f"  {'Sector':<28} {'Sharpe':>10} {'Spread (bps)':>14}")
    logger.info(f"  {'-'*55}")
    for sector, sharpe in ranked:
        spread = spreads.get(sector, 0) or 0
        flag = "STRONG" if (sharpe or 0) > 1.0 else ("WEAK" if (sharpe or 0) < 0 else "OK")
        logger.info(f"  {sector:<28} {(sharpe or 0):>10.3f} {spread:>14.1f}  {flag}")

    positive_sectors = sum(1 for v in sharpes.values() if (v or 0) > 0)
    total_sectors = len(sharpes)
    concentration = concentration or 0

    logger.info(f"\n  Positive sectors: {positive_sectors}/{total_sectors}")
    logger.info(f"  Concentration ratio: {concentration:.2f}  (gate: <= {PROMOTION_GATES['max_sector_concentration']})")
    passed = (
        positive_sectors >= (total_sectors * 0.5)
        and concentration <= PROMOTION_GATES["max_sector_concentration"]
    )
    logger.info(f"  Gate result: {'PASS' if passed else 'FAIL'}")
    return {
        "passed": passed,
        "positive_sectors": positive_sectors,
        "total_sectors": total_sectors,
        "concentration_ratio": concentration,
        "sector_sharpes": sharpes,
    }


def _check_core_metrics(agg: dict, profile: str) -> dict:
    _print_section(f"CORE METRICS: {profile}")
    nw_sharpe = agg.get("rank_long_short_sharpe_net_nw")
    ic = agg.get("ic_mean")
    auc = agg.get("overall_auc")
    mdd = agg.get("rank_max_drawdown_net")
    spread = (agg.get("rank_long_short_mean_net") or 0) * 10000

    logger.info(f"  WF AUC:        {auc:.4f}" if auc else "  WF AUC: N/A")
    logger.info(f"  IC mean:       {ic:.4f}" if ic else "  IC: N/A")
    logger.info(f"  NW Sharpe net: {nw_sharpe:.3f}" if nw_sharpe else "  NW Sharpe: N/A")
    logger.info(f"  Net spread:    {spread:.1f} bps/day")
    logger.info(f"  Max DD net:    {mdd:.3f}" if mdd else "  MDD: N/A")

    gates = {
        "nw_sharpe": (nw_sharpe or 0) >= PROMOTION_GATES["min_nw_sharpe_net"],
        "ic": (ic or 0) >= PROMOTION_GATES["min_ic"],
        "auc": (auc or 0) >= PROMOTION_GATES["min_wf_auc"],
        "mdd": (mdd or 0) >= PROMOTION_GATES["max_mdd_net"],
    }
    passed = all(gates.values())
    for g, ok in gates.items():
        logger.info(f"  Gate [{g}]: {'PASS' if ok else 'FAIL'}")
    return {"passed": passed, "nw_sharpe": nw_sharpe, "ic": ic, "auc": auc, "mdd": mdd}


def _run_profile(profile: str, df: pd.DataFrame) -> dict:
    logger.info(f"\nTraining profile: {profile}")
    result = train_baseline(
        df,
        experiment_name="research-promotion-readiness",
        run_name=f"promo_check_{profile}",
        dataset_version="v1.0-backfill",
        feature_profile=profile,
        run_mode="research",
    )
    bt = walk_forward_backtest(df, feature_profile=profile, **BT_KW)
    agg = bt.get("aggregate_metrics", {})
    return {"train_result": result, "agg": agg}


def main():
    logger.info("Loading dataset...")
    db = SessionLocal()
    end_date = date.today()
    start_date = end_date - timedelta(days=3 * 365)
    symbols = [row.symbol for row in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]
    df = build_training_dataset(db, symbols, start_date, end_date, **DATASET_KW)
    db.close()
    logger.info(f"Dataset: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
                f"{df['target_session_date'].nunique()} dates")

    # --- flow_enhanced: primary candidate ---
    fe_data = _run_profile("flow_enhanced", df)
    fe_agg = fe_data["agg"]

    core_check = _check_core_metrics(fe_agg, "flow_enhanced")
    time_check = _check_time_stability(fe_agg)
    sector_check = _check_sector_contribution(fe_agg)

    # --- flow_residual: residual flow ablation ---
    _print_section("CHECK 3: RESIDUAL FLOW ABLATION")
    fr_data = _run_profile("flow_residual", df)
    fr_agg = fr_data["agg"]
    fr_nw = fr_agg.get("rank_long_short_sharpe_net_nw")
    fr_ic = fr_agg.get("ic_mean")
    fe_nw = fe_agg.get("rank_long_short_sharpe_net_nw")
    fe_ic = fe_agg.get("ic_mean")
    logger.info(f"  {'Profile':<20} {'NW Sharpe':>12} {'IC':>10}")
    logger.info(f"  {'-'*44}")
    logger.info(f"  {'flow_enhanced':<20} {(fe_nw or 0):>12.3f} {(fe_ic or 0):>10.4f}")
    logger.info(f"  {'flow_residual':<20} {(fr_nw or 0):>12.3f} {(fr_ic or 0):>10.4f}")
    residual_delta = (fr_nw or 0) - (fe_nw or 0)
    logger.info(f"\n  Delta (residual - enhanced): {residual_delta:+.3f}")
    if residual_delta > 0.1:
        logger.info("  INSIGHT: Residual flow adds alpha. Consider adopting flow_residual as primary.")
    else:
        logger.info("  INSIGHT: Raw flow performs comparably. flow_enhanced remains preferred.")

    # --- Final promotion verdict ---
    _print_section("PROMOTION VERDICT")
    all_passed = core_check["passed"] and time_check["passed"] and sector_check["passed"]
    checks = {
        "core_metrics": core_check["passed"],
        "time_stability": time_check["passed"],
        "sector_breadth": sector_check["passed"],
    }
    for name, ok in checks.items():
        logger.info(f"  [{name}]: {'PASS' if ok else 'FAIL'}")
    logger.info(f"\n  {'>>> PROMOTE' if all_passed else '>>> NOT READY - HOLD AS CANDIDATE'}")
    if all_passed:
        logger.info("  Suggested command:")
        logger.info("    python -m scripts.train_and_promote \\")
        logger.info("      --dataset-version v1.0-backfill --target-mode sector_relative \\")
        logger.info("      --target-horizon-days 5 --include-liquidity-features \\")
        logger.info("      --feature-profile flow_enhanced --rank-top-n 20 \\")
        logger.info("      --transaction-cost-bps 10 --rank-sharpe-nw-lag 4 --allow-promote")

    # Save results
    out_dir = "artifacts/research"
    os.makedirs(out_dir, exist_ok=True)
    results = {
        "core": core_check,
        "time_stability": time_check,
        "sector_contribution": sector_check,
        "residual_flow_delta": residual_delta,
        "promote": all_passed,
    }
    with open(f"{out_dir}/promotion_readiness.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_dir}/promotion_readiness.json")


if __name__ == "__main__":
    main()
