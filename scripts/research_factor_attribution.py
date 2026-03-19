"""Fama-French 5-factor + Momentum attribution of the L/S equity curve.

Answers the existential question: is TalPred AI generating genuine alpha,
or is the strategy just replicating known systematic factor exposures
(market, size, value, profitability, investment, momentum)?

Methodology:
  1. Run walk-forward backtest to produce daily L/S returns.
  2. Download FF5 + MOM daily factor returns from Ken French's data library.
  3. Regress daily L/S returns on the 6 factors.
  4. Report: annualized alpha, t-stat, R², factor loadings, residual IC.

Interpretation:
  - R² > 0.70 → strategy is mostly factor replication
  - R² < 0.30 → strategy has meaningful idiosyncratic component
  - Alpha t-stat > 2.0 → residual alpha is statistically significant
  - Alpha t-stat < 1.0 → no evidence of genuine alpha

Usage:
  python -m scripts.research_factor_attribution
  python -m scripts.research_factor_attribution --years 5
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.db import SessionLocal
from src.features.engine import load_training_universe
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FF5_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Momentum_Factor_daily_CSV.zip"
)


def _download_ff_csv(url: str) -> pd.DataFrame:
    """Download and parse a Ken French CSV zip file."""
    logger.info(f"Downloading {url.split('/')[-1]} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw = zf.read(csv_name).decode("utf-8", errors="replace")

    lines = raw.strip().split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and len(stripped.split(",")[0].strip()) == 8:
            header_idx = i - 1
            break

    if header_idx is None:
        for i, line in enumerate(lines):
            if "Mkt-RF" in line or "Mom" in line:
                header_idx = i
                break

    if header_idx is None:
        raise ValueError("Could not locate header row in FF CSV")

    end_idx = len(lines)
    for i in range(header_idx + 2, len(lines)):
        stripped = lines[i].strip()
        if not stripped or (stripped and not stripped[0].isdigit()):
            end_idx = i
            break

    csv_text = "\n".join(lines[header_idx:end_idx])
    df = pd.read_csv(io.StringIO(csv_text))

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date_str"})
    df["date_str"] = df["date_str"].astype(str).str.strip()
    df = df[df["date_str"].str.match(r"^\d{8}$")]
    df["date"] = pd.to_datetime(df["date_str"], format="%Y%m%d")

    for col in df.columns:
        if col not in ("date_str", "date"):
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df = df.drop(columns=["date_str"]).set_index("date")
    return df


def load_ff5_mom() -> pd.DataFrame:
    """Load FF5 + Momentum daily factors, merged on date."""
    ff5 = _download_ff_csv(FF5_MOM_URL)
    mom = _download_ff_csv(MOM_URL)

    # Standardize column names
    rename_map = {}
    for col in ff5.columns:
        clean = col.strip().replace(" ", "_").replace("-", "_")
        rename_map[col] = clean
    ff5 = ff5.rename(columns=rename_map)

    rename_map_mom = {}
    for col in mom.columns:
        clean = col.strip().replace(" ", "_").replace("-", "_")
        if clean.upper() in ("MOM", "WML"):
            rename_map_mom[col] = "MOM"
        else:
            rename_map_mom[col] = clean
    mom = mom.rename(columns=rename_map_mom)

    if "MOM" not in mom.columns:
        first_numeric = [c for c in mom.columns if c not in ("RF",)][0]
        mom = mom.rename(columns={first_numeric: "MOM"})

    factors = ff5.join(mom[["MOM"]], how="inner")
    return factors


def run_backtest_for_ls_returns(
    db,
    years: int = 5,
    feature_profile: str = "cross_sectional_alpha",
    target_horizon_days: int = 5,
) -> pd.DataFrame:
    """Run walk-forward and extract the daily L/S net return series."""
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)

    symbols = load_training_universe(db, start_date, end_date)
    if not symbols:
        from src.models.schema import Symbol
        symbols = [r.symbol for r in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]
        logger.warning("PIT universe empty, using currently active symbols")

    logger.info(f"Building dataset: {len(symbols)} symbols, {start_date} to {end_date}")
    df = build_training_dataset(
        db, symbols, start_date, end_date,
        target_mode="market_relative",
        target_horizon_days=target_horizon_days,
        include_liquidity_features=True,
        sample_stride=target_horizon_days,
    )
    logger.info(f"Dataset: {len(df)} rows")

    logger.info("Running walk-forward backtest...")
    bt = walk_forward_backtest(
        df,
        min_train_days=252,
        step_days=21,
        rank_top_n=20,
        transaction_cost_bps=10.0,
        rank_rebalance_stride=target_horizon_days,
        rank_sharpe_nw_lag=max(0, target_horizon_days - 1),
        feature_profile=feature_profile,
    )
    if "error" in bt:
        raise RuntimeError(f"Backtest failed: {bt['error']}")

    agg = bt["aggregate_metrics"]
    series = agg.get("rank_daily_series", [])
    if not series:
        raise RuntimeError("No daily L/S return series in backtest output")

    ls_df = pd.DataFrame(series)
    ls_df["date"] = pd.to_datetime(ls_df["date"])
    ls_df = ls_df.set_index("date").sort_index()

    logger.info(
        f"Backtest complete: {len(ls_df)} days, "
        f"Sharpe_net={agg.get('rank_long_short_sharpe_net'):.3f}, "
        f"IC={agg.get('ic_mean')}, "
        f"DSR={agg.get('rank_deflated_sharpe_ratio')}"
    )
    return ls_df, agg


def run_factor_regression(
    ls_returns: pd.Series,
    factors: pd.DataFrame,
    annualization: float = 252.0,
) -> dict:
    """Regress L/S returns on FF5+MOM factors via OLS."""
    from scipy import stats as sp_stats

    merged = pd.DataFrame({"strategy": ls_returns}).join(factors, how="inner").dropna()

    if len(merged) < 30:
        return {"error": f"Only {len(merged)} overlapping dates, need >= 30"}

    y = merged["strategy"].values

    factor_names = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    available_factors = [f for f in factor_names if f in merged.columns]
    if not available_factors:
        return {"error": f"No factors found. Columns: {list(merged.columns)}"}

    X = merged[available_factors].values
    X_const = np.column_stack([np.ones(len(X)), X])

    # OLS: β = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix in OLS"}

    residuals = y - X_const @ beta
    n, k = len(y), X_const.shape[1]
    dof = n - k

    if dof <= 0:
        return {"error": f"Insufficient degrees of freedom (n={n}, k={k})"}

    sigma2 = np.sum(residuals ** 2) / dof
    cov_matrix = sigma2 * np.linalg.inv(X_const.T @ X_const)
    se = np.sqrt(np.diag(cov_matrix))
    t_stats = beta / se
    p_values = [float(2 * (1 - sp_stats.t.cdf(abs(t), dof))) for t in t_stats]

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / dof if dof > 0 else 0.0

    # Annualized alpha
    daily_alpha = beta[0]
    annual_alpha = daily_alpha * annualization
    alpha_t_stat = t_stats[0]
    alpha_p_value = p_values[0]

    # Residual analysis
    residual_std = float(np.std(residuals, ddof=1))
    residual_sharpe = float((np.mean(residuals) / residual_std) * np.sqrt(annualization)) \
        if residual_std > 0 else 0.0

    # Factor loadings
    loadings = {}
    for i, fname in enumerate(available_factors):
        loadings[fname] = {
            "beta": round(float(beta[i + 1]), 4),
            "t_stat": round(float(t_stats[i + 1]), 2),
            "p_value": round(float(p_values[i + 1]), 4),
        }

    return {
        "n_observations": n,
        "r_squared": round(r_squared, 4),
        "adj_r_squared": round(adj_r_squared, 4),
        "annual_alpha_pct": round(annual_alpha * 100, 3),
        "alpha_daily": round(daily_alpha, 6),
        "alpha_t_stat": round(float(alpha_t_stat), 2),
        "alpha_p_value": round(alpha_p_value, 4),
        "alpha_significant_5pct": alpha_p_value < 0.05,
        "residual_sharpe": round(residual_sharpe, 3),
        "residual_vol_annual_pct": round(residual_std * np.sqrt(annualization) * 100, 2),
        "factor_loadings": loadings,
        "factors_used": available_factors,
    }


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fama-French factor attribution")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--feature-profile", default="cross_sectional_alpha")
    parser.add_argument("--target-horizon-days", type=int, default=5)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        # Step 1: Get L/S returns from walk-forward
        ls_df, agg = run_backtest_for_ls_returns(
            db,
            years=args.years,
            feature_profile=args.feature_profile,
            target_horizon_days=args.target_horizon_days,
        )

        # Step 2: Download FF5 + MOM factors
        factors = load_ff5_mom()
        logger.info(f"Factor data: {factors.index.min().date()} to {factors.index.max().date()}")
        logger.info(f"Factor columns: {list(factors.columns)}")

        # Step 3: Run regression
        logger.info("Running factor regression (FF5 + MOM)...")
        result = run_factor_regression(ls_df["long_short_net"], factors)

        if "error" in result:
            logger.error(f"Regression failed: {result['error']}")
            return

        # Step 4: Print results
        print("\n" + "=" * 70)
        print("FAMA-FRENCH 5-FACTOR + MOMENTUM ATTRIBUTION")
        print("=" * 70)
        print(f"\nObservations:        {result['n_observations']}")
        print(f"R²:                  {result['r_squared']:.4f}")
        print(f"Adjusted R²:         {result['adj_r_squared']:.4f}")
        print(f"\nAnnualized Alpha:    {result['annual_alpha_pct']:+.3f}%")
        print(f"Alpha t-stat:        {result['alpha_t_stat']:.2f}")
        print(f"Alpha p-value:       {result['alpha_p_value']:.4f}")
        print(f"Alpha significant:   {'YES ✓' if result['alpha_significant_5pct'] else 'NO ✗'}")
        print(f"\nResidual Sharpe:     {result['residual_sharpe']:.3f}")
        print(f"Residual Vol (ann):  {result['residual_vol_annual_pct']:.2f}%")

        print(f"\n{'Factor':<12} {'Beta':>8} {'t-stat':>8} {'p-value':>8}")
        print("-" * 40)
        for fname, fdata in result["factor_loadings"].items():
            sig = "*" if fdata["p_value"] < 0.05 else " "
            print(f"{fname:<12} {fdata['beta']:>8.4f} {fdata['t_stat']:>8.2f} {fdata['p_value']:>7.4f} {sig}")

        # Interpretation
        print("\n" + "-" * 70)
        print("INTERPRETATION")
        print("-" * 70)

        r2 = result["r_squared"]
        if r2 > 0.70:
            print(f"R²={r2:.2f} → HIGH: Strategy is largely factor replication.")
            print("   The L/S returns are mostly explained by known systematic factors.")
            print("   Recommendation: This is factor beta, not alpha.")
        elif r2 > 0.30:
            print(f"R²={r2:.2f} → MODERATE: Mix of factor exposure and idiosyncratic signal.")
            print("   Some factor content, but there may be genuine alpha.")
        else:
            print(f"R²={r2:.2f} → LOW: Strategy has strong idiosyncratic component.")
            print("   Known factors explain little of the L/S returns.")

        t = result["alpha_t_stat"]
        if abs(t) > 2.0:
            print(f"Alpha t-stat={t:.2f} → Statistically significant residual alpha.")
        elif abs(t) > 1.5:
            print(f"Alpha t-stat={t:.2f} → Marginal significance. More data needed.")
        else:
            print(f"Alpha t-stat={t:.2f} → Not significant. Alpha may be zero.")

        # DSR from backtest
        dsr = agg.get("rank_deflated_sharpe_ratio")
        if dsr is not None:
            print(f"\nDeflated Sharpe Ratio: {dsr:.4f}")
            if dsr > 0.95:
                print("   DSR > 0.95 → Sharpe survives multiple testing correction.")
            elif dsr > 0.50:
                print("   DSR 0.50-0.95 → Borderline. May or may not survive scrutiny.")
            else:
                print("   DSR < 0.50 → Sharpe likely explained by data mining.")

        # Save results
        out_dir = Path("artifacts") / "research"
        out_dir.mkdir(parents=True, exist_ok=True)

        import json
        out_json = out_dir / "factor_attribution.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {out_json}")

        # Generate plot
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Factor loadings bar chart
            fnames = list(result["factor_loadings"].keys())
            betas = [result["factor_loadings"][f]["beta"] for f in fnames]
            colors = ["#e74c3c" if b < 0 else "#2ecc71" for b in betas]
            axes[0].barh(fnames, betas, color=colors)
            axes[0].set_xlabel("Factor Loading (β)")
            axes[0].set_title("Factor Exposures")
            axes[0].axvline(x=0, color="black", linewidth=0.5)
            axes[0].grid(True, alpha=0.3)

            # Cumulative returns: strategy vs explained vs residual
            merged = pd.DataFrame({"strategy": ls_df["long_short_net"]}).join(factors, how="inner").dropna()
            avail = [f for f in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"] if f in merged.columns]
            factor_return = sum(
                result["factor_loadings"][f]["beta"] * merged[f]
                for f in avail if f in result["factor_loadings"]
            )
            residual = merged["strategy"] - factor_return - result["alpha_daily"]

            cum_strat = (1 + merged["strategy"]).cumprod()
            cum_factor = (1 + factor_return).cumprod()
            cum_alpha = (1 + merged["strategy"] - factor_return).cumprod()

            axes[1].plot(cum_strat.index, cum_strat.values, label="Strategy (total)", linewidth=1.5)
            axes[1].plot(cum_factor.index, cum_factor.values, label=f"Factor component (R²={r2:.2f})", linewidth=1.2, linestyle="--")
            axes[1].plot(cum_alpha.index, cum_alpha.values, label="Residual (alpha)", linewidth=1.2, linestyle=":")
            axes[1].set_ylabel("Cumulative Return")
            axes[1].set_title("Return Decomposition")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

            plt.suptitle(
                f"FF5+MOM Attribution — α={result['annual_alpha_pct']:+.2f}% (t={result['alpha_t_stat']:.1f}), R²={r2:.2f}",
                fontsize=12,
            )
            plt.tight_layout()

            out_png = out_dir / "factor_attribution.png"
            plt.savefig(out_png, dpi=160)
            plt.close()
            print(f"Plot saved to: {out_png}")
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
