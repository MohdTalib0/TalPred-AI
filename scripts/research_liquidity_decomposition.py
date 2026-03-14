"""Liquidity-factor decomposition and size-neutral diagnostics."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol
from src.models.trainer import train_baseline


LIQUIDITY_COLS = [
    "log_market_cap",
    "market_cap_rank",
    "dollar_volume",
    "dollar_volume_rank_market",
    "turnover_ratio",
    "volume_change_5d",
    "volume_zscore_20d",
    "volatility_expansion_5_20",
]

SIZE_COLS = ["log_market_cap", "market_cap_rank"]
TURNOVER_COLS = ["turnover_ratio"]
DOLLAR_VOLUME_COLS = ["dollar_volume", "dollar_volume_rank_market"]
VOLUME_SHOCK_COLS = ["volume_change_5d", "volume_zscore_20d", "volatility_expansion_5_20"]

# Keep regime_label so one-hot regime dummies still work.
MINIMAL_KEEP_COLS = ["symbol", "target_session_date", "direction", "target_value", "next_day_return", "regime_label"]


def _evaluate(df: pd.DataFrame, label: str) -> dict:
    tr = train_baseline(df, run_name=f"research_liq_decomp_{label}", dataset_version="v1.0-backfill")
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
    return {
        "config": label,
        "val_auc": tr["metrics"]["auc_roc"],
        "wf_auc": agg.get("overall_auc"),
        "ic_mean": agg.get("ic_mean"),
        "rank_net_sharpe_nw": agg.get("rank_long_short_sharpe_net_nw"),
        "rank_net_mean": agg.get("rank_long_short_mean_net"),
        "decile_sharpe": agg.get("decile_spread_sharpe"),
    }


def _liquidity_only_df(base_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in (MINIMAL_KEEP_COLS + LIQUIDITY_COLS) if c in base_df.columns]
    return base_df[keep_cols].copy()


def _drop_cols(base_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return base_df.drop(columns=[c for c in cols if c in base_df.columns], errors="ignore").copy()


def _size_neutralize_liquidity(liq_df: pd.DataFrame) -> pd.DataFrame:
    """Replace liquidity features by within-size-bucket ranks per date."""
    df = liq_df.copy()
    if "market_cap_rank" not in df.columns:
        return df

    # 5 size buckets by market_cap_rank, per day.
    df["size_bucket"] = (
        (np.floor(df["market_cap_rank"].clip(lower=0, upper=0.9999) * 5)).astype(int)
    )

    rank_cols = [c for c in LIQUIDITY_COLS if c in df.columns and c not in {"log_market_cap", "market_cap_rank"}]
    for col in rank_cols:
        df[col] = df.groupby(["target_session_date", "size_bucket"])[col].rank(pct=True)

    # Keep size columns too; the ranking transform reduces raw size leakage in liquidity signals.
    df = df.drop(columns=["size_bucket"])
    return df


def main() -> None:
    load_dotenv()
    db = SessionLocal()
    try:
        symbols = [r.symbol for r in db.query(Symbol).filter(Symbol.is_active.is_(True)).all()]
        end = date.today()
        start = end - timedelta(days=3 * 365)

        base_df = build_training_dataset(
            db,
            symbols,
            start,
            end,
            target_mode="sector_relative",
            target_horizon_days=5,
            include_liquidity_features=True,
        )

        experiments: list[tuple[str, pd.DataFrame]] = []
        experiments.append(("baseline_all_features", base_df))
        experiments.append(("liquidity_only", _liquidity_only_df(base_df)))
        experiments.append(("drop_size_only", _drop_cols(base_df, SIZE_COLS)))
        experiments.append(("drop_turnover_only", _drop_cols(base_df, TURNOVER_COLS)))
        experiments.append(("drop_dollar_volume_only", _drop_cols(base_df, DOLLAR_VOLUME_COLS)))
        experiments.append(("drop_volume_shocks_only", _drop_cols(base_df, VOLUME_SHOCK_COLS)))
        liq_only = _liquidity_only_df(base_df)
        experiments.append(("liquidity_only_size_neutral", _size_neutralize_liquidity(liq_only)))

        rows = []
        for label, df in experiments:
            rows.append(_evaluate(df, label))

        out_df = pd.DataFrame(rows)
        base_row = out_df[out_df["config"] == "baseline_all_features"].iloc[0]
        for metric in ["wf_auc", "ic_mean", "rank_net_sharpe_nw", "decile_sharpe"]:
            out_df[f"delta_{metric}"] = out_df[metric] - base_row[metric]

        out_dir = Path("artifacts") / "research"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "liquidity_decomposition_summary.csv"
        out_rank_csv = out_dir / "liquidity_decomposition_ranked.csv"
        out_df.to_csv(out_csv, index=False)
        out_df.sort_values("delta_rank_net_sharpe_nw").to_csv(out_rank_csv, index=False)

        print(f"LIQ_DECOMP_SUMMARY_CSV={out_csv}")
        print(f"LIQ_DECOMP_RANKED_CSV={out_rank_csv}")
        print(out_df.sort_values('delta_rank_net_sharpe_nw').to_string(index=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
