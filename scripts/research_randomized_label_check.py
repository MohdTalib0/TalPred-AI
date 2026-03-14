"""Randomized-label sanity check for leakage/artifact detection."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from dotenv import load_dotenv

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol
from src.models.trainer import train_baseline


def _shuffle_within_date(df):
    shuffled = df.copy()
    rng = np.random.default_rng(42)
    shuffled["direction"] = (
        shuffled.groupby("target_session_date")["direction"]
        .transform(lambda s: s.iloc[rng.permutation(len(s))].to_numpy())
        .astype(int)
    )
    return shuffled


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
        df_shuf = _shuffle_within_date(df)

        train_res = train_baseline(
            df_shuf,
            run_name="research_randomized_labels_h5",
            dataset_version="v1.0-backfill",
        )
        bt = walk_forward_backtest(
            df_shuf,
            min_train_days=252,
            step_days=21,
            rank_top_n=20,
            rank_mode="global",
            transaction_cost_bps=10,
            rank_rebalance_stride=1,
            rank_sharpe_nw_lag=4,
        )
        agg = bt["aggregate_metrics"]
        print("RANDOMIZED_LABEL_SANITY_CHECK")
        print(
            f"VAL_AUC={train_res['metrics']['auc_roc']:.4f} "
            f"WF_AUC={agg.get('overall_auc'):.4f} "
            f"NET_SHARPE={agg.get('rank_long_short_sharpe_net'):.3f} "
            f"NET_SHARPE_NW={agg.get('rank_long_short_sharpe_net_nw'):.3f} "
            f"NET_SPREAD={agg.get('rank_long_short_mean_net'):.6f}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
