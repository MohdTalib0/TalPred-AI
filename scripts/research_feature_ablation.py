"""Feature-family ablation study for signal attribution."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.db import SessionLocal
from src.features.leakage import build_training_dataset
from src.models.backtest import walk_forward_backtest
from src.models.schema import Symbol
from src.models.trainer import train_baseline


FAMILY_MAP: dict[str, list[str]] = {
    "momentum_technical": [
        "rsi_14",
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "momentum_60d",
        "momentum_120d",
        "rolling_return_5d",
        "rolling_return_20d",
        "rolling_volatility_20d",
        "macd",
        "macd_signal",
        "short_term_reversal",
    ],
    "cross_sectional_ranks": [
        "momentum_rank_market",
        "momentum_60d_rank_market",
        "momentum_120d_rank_market",
        "short_term_reversal_rank_market",
        "volatility_rank_market",
        "rsi_rank_market",
        "volume_rank_market",
        "sector_momentum_rank",
    ],
    "liquidity_size": [
        "volume_change_5d",
        "volume_zscore_20d",
        "volatility_expansion_5_20",
        "log_market_cap",
        "market_cap_rank",
        "dollar_volume",
        "turnover_ratio",
        "dollar_volume_rank_market",
    ],
    "sector_benchmark": [
        "sector_return_1d",
        "sector_return_5d",
        "sector_relative_return_1d",
        "sector_relative_return_5d",
        "benchmark_relative_return_1d",
    ],
    "news_nlp": [
        "news_sentiment_24h",
        "news_sentiment_7d",
        "news_sentiment_std",
        "news_positive_ratio",
        "news_negative_ratio",
        "news_volume",
        "news_credibility_avg",
        "news_present_flag",
    ],
    "macro_regime": [
        "vix_level",
        "sp500_momentum_200d",
        "regime_label",
    ],
}


def _run_config(df: pd.DataFrame, label: str) -> dict:
    train_res = train_baseline(
        df,
        run_name=f"research_ablation_{label}",
        dataset_version="v1.0-backfill",
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
    return {
        "config": label,
        "val_auc": train_res["metrics"]["auc_roc"],
        "wf_auc": agg.get("overall_auc"),
        "ic_mean": agg.get("ic_mean"),
        "rank_net_sharpe_nw": agg.get("rank_long_short_sharpe_net_nw"),
        "rank_net_mean": agg.get("rank_long_short_mean_net"),
        "decile_sharpe": agg.get("decile_spread_sharpe"),
        "decile_monotonicity": agg.get("decile_monotonicity_spearman"),
    }


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

        rows: list[dict] = []
        rows.append(_run_config(base_df, "baseline_all_features"))

        for family, cols in FAMILY_MAP.items():
            drop_cols = [c for c in cols if c in base_df.columns]
            ab_df = base_df.drop(columns=drop_cols, errors="ignore")
            rows.append(_run_config(ab_df, f"drop_{family}"))

        out_dir = Path("artifacts") / "research"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "feature_ablation_summary.csv"
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_csv, index=False)

        base = out_df[out_df["config"] == "baseline_all_features"].iloc[0]
        for metric in ["wf_auc", "ic_mean", "rank_net_sharpe_nw", "decile_sharpe"]:
            out_df[f"delta_{metric}"] = out_df[metric] - base[metric]

        out_rank = out_df.sort_values("delta_rank_net_sharpe_nw")
        out_rank_csv = out_dir / "feature_ablation_ranked_by_sharpe_impact.csv"
        out_rank.to_csv(out_rank_csv, index=False)

        print(f"ABLATION_SUMMARY_CSV={out_csv}")
        print(f"ABLATION_RANKED_CSV={out_rank_csv}")
        print(out_rank.to_string(index=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
