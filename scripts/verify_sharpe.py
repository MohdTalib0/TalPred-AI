"""
Stride-5 backtest to verify Sharpe is NOT inflated by overlapping returns,
plus regime stability diagnostics.
"""
import sys
import logging
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    stream=sys.stdout,
)

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol


def main():
    db = SessionLocal()
    symbols = [
        r.symbol
        for r in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()
    ]
    end_date = date.today()
    start_date = end_date - timedelta(days=3 * 365)

    print("Building dataset (market_relative, 5-day horizon)...")
    df = build_training_dataset(
        db,
        symbols,
        start_date,
        end_date,
        target_mode="market_relative",
        include_liquidity_features=True,
        target_horizon_days=5,
    )
    print(f"Dataset: {len(df):,} rows")

    print()
    print("=" * 60)
    print("STRIDE=5 BACKTEST  (non-overlapping returns => true Sharpe)")
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
        db.close()
        return

    a = bt["aggregate_metrics"]

    print(f"  IC mean:            {a.get('ic_mean')}")
    print(f"  IC std:             {a.get('ic_std')}")
    print(f"  IC IR:              {a.get('ic_ir')}")
    print(f"  Sharpe (net):       {a.get('rank_long_short_sharpe_net')}")
    print(f"  Sharpe (net, NW):   {a.get('rank_long_short_sharpe_net_nw')}")
    print(f"  Max Drawdown (net): {a.get('rank_max_drawdown_net')}")
    print(f"  Decile Mono:        {a.get('decile_monotonicity_spearman')}")
    print(f"  Avg Turnover:       {a.get('rank_avg_turnover')}")
    print(f"  Days:               {a.get('rank_days')}")

    print()
    print("--- Yearly Sharpe ---")
    yearly = a.get("rank_yearly_sharpe_net", {})
    for yr, val in sorted(yearly.items()):
        print(f"  {yr}: {val}")

    print()
    print("--- Regime Stability ---")
    print(f"  IC (bull/low-vol):  {a.get('regime_ic_bull_low_vol')}")
    print(f"  IC (bear/high-vol): {a.get('regime_ic_bear_high_vol')}")
    print(f"  IC (sideways):      {a.get('regime_ic_sideways')}")
    print(f"  IC~VIX corr:        {a.get('ic_vix_corr')}")
    print(f"  Dispersion mean:    {a.get('dispersion_mean')}")
    print(f"  IC~Disp corr:       {a.get('ic_dispersion_corr')}")

    # stride=1 for comparison
    print()
    print("=" * 60)
    print("STRIDE=1 BACKTEST  (overlapping returns => inflated Sharpe?)")
    print("=" * 60)
    bt1 = walk_forward_backtest(
        df,
        min_train_days=252,
        step_days=21,
        rank_top_n=20,
        transaction_cost_bps=10.0,
        rank_rebalance_stride=1,
        feature_profile="cross_sectional_alpha",
    )
    if "error" not in bt1:
        a1 = bt1["aggregate_metrics"]
        print(f"  Sharpe (net):       {a1.get('rank_long_short_sharpe_net')}")
        print(f"  Sharpe (net, NW):   {a1.get('rank_long_short_sharpe_net_nw')}")
        print(f"  IC mean:            {a1.get('ic_mean')}")
        print(f"  IC IR:              {a1.get('ic_ir')}")
    else:
        print(f"  ERROR: {bt1['error']}")

    print()
    print("=" * 60)
    ratio_msg = "Sharpe ratio: stride1/stride5"
    try:
        s1 = float(a1.get("rank_long_short_sharpe_net", 0))
        s5 = float(a.get("rank_long_short_sharpe_net", 0))
        if s5 != 0:
            print(f"  {ratio_msg} = {s1/s5:.2f}x")
            if s1 / s5 > 2.0:
                print("  WARNING: stride=1 Sharpe >2x stride=5 => overlapping return inflation!")
            else:
                print("  OK: ratio <= 2x => Sharpe is real.")
        else:
            print(f"  {ratio_msg} = N/A (stride5 Sharpe is 0)")
    except Exception as e:
        print(f"  Could not compute ratio: {e}")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
