"""Walk-forward backtest pipeline (ML-206).

Implements expanding-window walk-forward validation to evaluate model
performance without look-ahead bias.
"""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.trainer import DEFAULT_PARAMS, prepare_features

logger = logging.getLogger(__name__)


def walk_forward_backtest(
    df: pd.DataFrame,
    min_train_days: int = 252,
    step_days: int = 21,
    params: dict | None = None,
    rank_top_n: int = 20,
    rank_mode: str = "global",
    rank_per_sector_n: int = 2,
    transaction_cost_bps: float = 0.0,
    rank_rebalance_stride: int = 1,
    rank_sharpe_nw_lag: int = 0,
    feature_profile: str = "all_features",
    rank_weight_mode: str = "equal",
) -> dict:
    """Run walk-forward expanding-window backtest.

    Args:
        df: Full dataset with features + direction + target_session_date
        min_train_days: Minimum trading days before first prediction
        step_days: Number of days per evaluation step
        params: XGBoost parameters

    Returns dict with per-step metrics and aggregate results.
    """
    params = params or DEFAULT_PARAMS.copy()
    df = df.sort_values("target_session_date").reset_index(drop=True)

    dates = sorted(df["target_session_date"].unique())
    if len(dates) < min_train_days + step_days:
        logger.warning(f"Not enough data for backtest: {len(dates)} dates, need {min_train_days + step_days}")
        return {"error": "insufficient data"}

    step_results = []
    all_predictions = []

    start_idx = min_train_days
    while start_idx < len(dates):
        end_idx = min(start_idx + step_days, len(dates))

        train_end = dates[start_idx - 1]
        test_start = dates[start_idx]
        test_end = dates[end_idx - 1]

        train_df = df[df["target_session_date"] <= train_end]
        test_df = df[(df["target_session_date"] >= test_start) & (df["target_session_date"] <= test_end)]

        if len(train_df) < 100 or len(test_df) < 10:
            start_idx += step_days
            continue

        X_train, y_train, train_medians = prepare_features(train_df, feature_profile=feature_profile)
        X_test, y_test, _ = prepare_features(test_df, fill_medians=train_medians, feature_profile=feature_profile)

        common_cols = [c for c in X_train.columns if c in X_test.columns]
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        step_acc = accuracy_score(y_test, y_pred)
        step_auc = roc_auc_score(y_test, y_prob) if len(y_test.unique()) > 1 else 0.5

        step_results.append({
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "accuracy": step_acc,
            "auc_roc": step_auc,
        })

        for i, (_, row) in enumerate(test_df.iterrows()):
            all_predictions.append({
                "symbol": row["symbol"],
                "sector": row["sector"] if "sector" in row else None,
                "regime_label": row["regime_label"] if "regime_label" in row else None,
                "vix_level": float(row["vix_level"]) if "vix_level" in row and pd.notna(row["vix_level"]) else np.nan,
                "date": row["target_session_date"],
                "actual": int(y_test.iloc[i]),
                "predicted": int(y_pred[i]),
                "probability_up": float(y_prob[i]),
                "target_value": float(row["target_value"]) if "target_value" in row and pd.notna(row["target_value"]) else np.nan,
                "next_day_return": float(row["next_day_return"]) if "next_day_return" in row and pd.notna(row["next_day_return"]) else np.nan,
            })

        logger.info(
            f"  Step {len(step_results)}: "
            f"train<={train_end}, test={test_start}→{test_end}, "
            f"acc={step_acc:.4f}, auc={step_auc:.4f}"
        )

        start_idx += step_days

    if not step_results:
        return {"error": "no valid steps"}

    preds_df = pd.DataFrame(all_predictions)
    agg_metrics = _compute_aggregate_metrics(
        preds_df,
        step_results,
        rank_top_n=rank_top_n,
        rank_mode=rank_mode,
        rank_per_sector_n=rank_per_sector_n,
        transaction_cost_bps=transaction_cost_bps,
        rank_rebalance_stride=rank_rebalance_stride,
        rank_sharpe_nw_lag=rank_sharpe_nw_lag,
        rank_weight_mode=rank_weight_mode,
    )

    logger.info(f"Walk-forward backtest complete:")
    logger.info(f"  Steps: {len(step_results)}")
    logger.info(f"  Overall accuracy: {agg_metrics['overall_accuracy']:.4f}")
    logger.info(f"  Overall AUC: {agg_metrics['overall_auc']:.4f}")
    logger.info(f"  Avg step accuracy: {agg_metrics['avg_step_accuracy']:.4f}")

    if agg_metrics["overall_accuracy"] > 0.65:
        logger.warning(
            f"LEAKAGE ALERT: Backtest accuracy {agg_metrics['overall_accuracy']:.4f} > 0.65"
        )

    return {
        "step_results": step_results,
        "predictions": preds_df,
        "aggregate_metrics": agg_metrics,
    }


def _compute_aggregate_metrics(
    preds_df: pd.DataFrame,
    step_results: list[dict],
    rank_top_n: int = 20,
    rank_mode: str = "global",
    rank_per_sector_n: int = 2,
    transaction_cost_bps: float = 0.0,
    rank_rebalance_stride: int = 1,
    rank_sharpe_nw_lag: int = 0,
    rank_weight_mode: str = "equal",
) -> dict:
    """Compute aggregate backtest metrics."""
    overall_acc = accuracy_score(preds_df["actual"], preds_df["predicted"])
    overall_auc = roc_auc_score(preds_df["actual"], preds_df["probability_up"]) \
        if preds_df["actual"].nunique() > 1 else 0.5

    step_accs = [s["accuracy"] for s in step_results]

    confident = preds_df[preds_df["probability_up"].apply(lambda p: max(p, 1 - p)) >= 0.6]
    confident_acc = accuracy_score(confident["actual"], confident["predicted"]) if len(confident) > 0 else None

    rank_metrics = _compute_ranking_metrics(
        preds_df,
        top_n=rank_top_n,
        rank_mode=rank_mode,
        rank_per_sector_n=rank_per_sector_n,
        transaction_cost_bps=transaction_cost_bps,
        rank_rebalance_stride=rank_rebalance_stride,
        rank_sharpe_nw_lag=rank_sharpe_nw_lag,
        rank_weight_mode=rank_weight_mode,
    )
    ic_metrics = _compute_ic_metrics(preds_df, rank_rebalance_stride=rank_rebalance_stride)
    rolling_ic_metrics = _compute_rolling_ic_metrics(preds_df, rank_rebalance_stride=rank_rebalance_stride, window=60)
    regime_ic_metrics = _compute_regime_ic_metrics(preds_df, rank_rebalance_stride=rank_rebalance_stride)
    decile_metrics = _compute_decile_factor_metrics(preds_df, rank_rebalance_stride=rank_rebalance_stride)
    monotonicity_metrics = _compute_decile_monotonicity_metrics(preds_df, rank_rebalance_stride=rank_rebalance_stride)
    prob_bin_metrics = _compute_probability_bin_metrics(preds_df, rank_rebalance_stride=rank_rebalance_stride)
    dispersion_metrics = _compute_dispersion_metrics(preds_df, rank_rebalance_stride=rank_rebalance_stride)

    return {
        "overall_accuracy": overall_acc,
        "overall_auc": overall_auc,
        "avg_step_accuracy": np.mean(step_accs),
        "std_step_accuracy": np.std(step_accs),
        "min_step_accuracy": np.min(step_accs),
        "max_step_accuracy": np.max(step_accs),
        "n_steps": len(step_results),
        "n_predictions": len(preds_df),
        "confident_accuracy": confident_acc,
        "confident_n": len(confident),
        "pct_up_predicted": (preds_df["predicted"] == 1).mean(),
        "pct_up_actual": (preds_df["actual"] == 1).mean(),
        **rank_metrics,
        **ic_metrics,
        **rolling_ic_metrics,
        **regime_ic_metrics,
        **decile_metrics,
        **monotonicity_metrics,
        **prob_bin_metrics,
        **dispersion_metrics,
        **_compute_sector_ranking_metrics(preds_df),
    }


def _compute_sector_ranking_metrics(
    preds_df: pd.DataFrame,
    top_n_per_sector: int = 3,
) -> dict:
    """Compute per-sector long/short Sharpe contribution.

    For each sector, runs a top_n/bottom_n equal-weight ranking independently.
    Reveals whether alpha is broad-based or concentrated in specific sectors.
    Returns a 'sector_sharpes' dict and a 'sector_concentration_ratio' flag.
    """
    if "sector" not in preds_df.columns or "target_value" not in preds_df.columns:
        return {}

    sectors = preds_df["sector"].dropna().unique()
    sector_sharpes: dict[str, float | None] = {}
    sector_spreads: dict[str, float | None] = {}
    sector_days: dict[str, int] = {}

    for sector in sectors:
        sec_df = preds_df[preds_df["sector"] == sector]
        daily = []
        for _, day_df in sec_df.groupby("date"):
            day = day_df.dropna(subset=["target_value"]).sort_values("probability_up", ascending=False)
            if len(day) < (2 * top_n_per_sector):
                continue
            long_leg = float(day.head(top_n_per_sector)["target_value"].mean())
            short_leg = float(day.tail(top_n_per_sector)["target_value"].mean())
            daily.append(long_leg - short_leg)

        if len(daily) < 20:
            continue

        arr = np.array(daily)
        mean_ret = float(arr.mean())
        std_ret = float(arr.std(ddof=1))
        sector_sharpes[str(sector)] = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else None
        sector_spreads[str(sector)] = round(mean_ret * 10000, 2)  # bps
        sector_days[str(sector)] = len(daily)

    if not sector_sharpes:
        return {"sector_sharpes": {}, "sector_spreads_bps": {}, "sector_days": {}, "sector_concentration_ratio": None}

    valid_sharpes = [v for v in sector_sharpes.values() if v is not None]
    concentration_ratio = (
        float(max(valid_sharpes)) / float(np.mean(valid_sharpes))
        if valid_sharpes and np.mean(valid_sharpes) != 0 else None
    )
    ranked = sorted(sector_sharpes.items(), key=lambda x: (x[1] or -999), reverse=True)

    return {
        "sector_sharpes": sector_sharpes,
        "sector_spreads_bps": sector_spreads,
        "sector_days": sector_days,
        "sector_sharpe_ranked": ranked,
        "sector_concentration_ratio": concentration_ratio,
    }


def _compute_ranking_metrics(
    preds_df: pd.DataFrame,
    top_n: int = 20,
    rank_mode: str = "global",
    rank_per_sector_n: int = 2,
    transaction_cost_bps: float = 0.0,
    rank_rebalance_stride: int = 1,
    rank_sharpe_nw_lag: int = 0,
    rank_weight_mode: str = "equal",
) -> dict:
    """Compute simple long/short ranking metrics from per-day prediction ranking."""
    if "target_value" not in preds_df.columns:
        return {
            "rank_long_short_mean": None,
            "rank_long_only_mean": None,
            "rank_long_short_sharpe": None,
            "rank_days": 0,
        }

    if rank_rebalance_stride < 1:
        raise ValueError("rank_rebalance_stride must be >= 1")
    if rank_sharpe_nw_lag < 0:
        raise ValueError("rank_sharpe_nw_lag must be >= 0")

    eval_df = preds_df
    if rank_rebalance_stride > 1:
        rebalance_dates = sorted(eval_df["date"].dropna().unique())[::rank_rebalance_stride]
        eval_df = eval_df[eval_df["date"].isin(rebalance_dates)]

    daily = []
    prev_long: set[str] | None = None
    prev_short: set[str] | None = None
    for dt, day_df in eval_df.groupby("date"):
        day = day_df.dropna(subset=["target_value"]).sort_values("probability_up", ascending=False)
        if rank_mode == "global":
            if len(day) < (2 * top_n):
                continue
            long_pick = day.head(top_n)
            short_pick = day.tail(top_n)
        elif rank_mode == "sector_neutral":
            if "sector" not in day.columns:
                continue
            long_parts = []
            short_parts = []
            for _, sec_df in day.groupby("sector"):
                if len(sec_df) < (2 * rank_per_sector_n):
                    continue
                sec_df = sec_df.sort_values("probability_up", ascending=False)
                long_parts.append(sec_df.head(rank_per_sector_n))
                short_parts.append(sec_df.tail(rank_per_sector_n))
            if not long_parts or not short_parts:
                continue
            long_pick = pd.concat(long_parts, ignore_index=True)
            short_pick = pd.concat(short_parts, ignore_index=True)
            top_n = len(long_pick)
        else:
            raise ValueError(f"Unsupported rank_mode: {rank_mode}")
        if rank_weight_mode == "signal":
            # Weight each position by its conviction: distance of p from 0.5.
            # Long: higher p → more weight. Short: lower p → more weight.
            long_scores = (long_pick["probability_up"] - 0.5).clip(lower=0)
            short_scores = (0.5 - short_pick["probability_up"]).clip(lower=0)
            long_w = long_scores / long_scores.sum() if long_scores.sum() > 0 else None
            short_w = short_scores / short_scores.sum() if short_scores.sum() > 0 else None
            long_leg = float((long_pick["target_value"] * long_w).sum()) if long_w is not None else float(long_pick["target_value"].mean())
            short_leg = float((short_pick["target_value"] * short_w).sum()) if short_w is not None else float(short_pick["target_value"].mean())
        else:
            long_leg = long_pick["target_value"].mean()
            short_leg = short_pick["target_value"].mean()
        long_syms = set(long_pick["symbol"].tolist())
        short_syms = set(short_pick["symbol"].tolist())
        turnover = None
        if prev_long is not None and prev_short is not None:
            long_turn = 1 - (len(prev_long & long_syms) / top_n)
            short_turn = 1 - (len(prev_short & short_syms) / top_n)
            turnover = (long_turn + short_turn) / 2
        # Two-sided daily cost approximation: long + short turnover
        cost = ((turnover or 0.0) * 2.0) * (transaction_cost_bps / 10000.0)
        net_long_short = float(long_leg - short_leg - cost)
        daily.append({
            "date": dt,
            "long_only": float(long_leg),
            "long_short": float(long_leg - short_leg),
            "long_short_net": net_long_short,
            "turnover": turnover,
            "cost": cost,
        })
        prev_long = long_syms
        prev_short = short_syms

    if not daily:
        return {
            "rank_long_short_mean": None,
            "rank_long_only_mean": None,
            "rank_long_short_sharpe": None,
            "rank_days": 0,
        }

    ddf = pd.DataFrame(daily)
    periods_per_year = 252.0 / float(rank_rebalance_stride)
    annualization = np.sqrt(periods_per_year)

    ls_mean = float(ddf["long_short"].mean())
    ls_mean_net = float(ddf["long_short_net"].mean())
    lo_mean = float(ddf["long_only"].mean())
    ls_std = float(ddf["long_short"].std())
    ls_std_net = float(ddf["long_short_net"].std())
    sharpe = float((ls_mean / ls_std) * annualization) if ls_std > 0 else None
    sharpe_net = float((ls_mean_net / ls_std_net) * annualization) if ls_std_net > 0 else None
    sharpe_nw = _newey_west_adjusted_sharpe(ddf["long_short"].to_numpy(), rank_sharpe_nw_lag, periods_per_year)
    sharpe_net_nw = _newey_west_adjusted_sharpe(ddf["long_short_net"].to_numpy(), rank_sharpe_nw_lag, periods_per_year)
    equity_curve = (1 + ddf["long_short"]).cumprod()
    equity_curve_net = (1 + ddf["long_short_net"]).cumprod()
    rolling_peak = equity_curve.cummax()
    rolling_peak_net = equity_curve_net.cummax()
    drawdown = (equity_curve / rolling_peak) - 1
    drawdown_net = (equity_curve_net / rolling_peak_net) - 1
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else None
    max_drawdown_net = float(drawdown_net.min()) if len(drawdown_net) > 0 else None
    avg_turnover = float(ddf["turnover"].dropna().mean()) if "turnover" in ddf.columns else None

    # Time-slice stability (yearly Sharpe on gross and net returns)
    yearly_gross = {}
    yearly_net = {}
    ddf["year"] = pd.to_datetime(ddf["date"]).dt.year
    for yr, grp in ddf.groupby("year"):
        std_g = float(grp["long_short"].std())
        std_n = float(grp["long_short_net"].std())
        yearly_gross[str(int(yr))] = float((grp["long_short"].mean() / std_g) * annualization) if std_g > 0 else None
        yearly_net[str(int(yr))] = float((grp["long_short_net"].mean() / std_n) * annualization) if std_n > 0 else None
    return {
        "rank_long_short_mean": ls_mean,
        "rank_long_short_mean_net": ls_mean_net,
        "rank_long_only_mean": lo_mean,
        "rank_long_short_sharpe": sharpe,
        "rank_long_short_sharpe_net": sharpe_net,
        "rank_long_short_sharpe_nw": sharpe_nw,
        "rank_long_short_sharpe_net_nw": sharpe_net_nw,
        "rank_max_drawdown": max_drawdown,
        "rank_max_drawdown_net": max_drawdown_net,
        "rank_avg_turnover": avg_turnover,
        "rank_avg_cost_daily": float(ddf["cost"].mean()),
        "rank_mode": rank_mode,
        "rank_top_n_effective": int(top_n),
        "rank_rebalance_stride": rank_rebalance_stride,
        "rank_sharpe_nw_lag": rank_sharpe_nw_lag,
        "rank_yearly_sharpe_gross": yearly_gross,
        "rank_yearly_sharpe_net": yearly_net,
        "rank_days": len(ddf),
        # Raw daily return series for downstream rolling analysis.
        # List of (date_str, gross_ls, net_ls) sorted by date.
        "rank_daily_series": [
            {
                "date": str(r["date"]),
                "long_short": r["long_short"],
                "long_short_net": r["long_short_net"],
            }
            for r in daily
        ],
    }


def _newey_west_adjusted_sharpe(returns: np.ndarray, lag: int, periods_per_year: float = 252.0) -> float | None:
    """Compute Lo/Newey-West adjusted annualized Sharpe for serially correlated returns."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return None
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return None
    base_sr = float((np.mean(arr) / std) * np.sqrt(periods_per_year))
    if lag <= 0:
        return base_sr

    x = arr - float(np.mean(arr))
    var0 = float(np.dot(x, x) / (n - 1))
    if var0 <= 0:
        return base_sr

    max_lag = min(lag, n - 1)
    rho_sum = 0.0
    for k in range(1, max_lag + 1):
        gamma_k = float(np.dot(x[k:], x[:-k]) / (n - 1))
        rho_k = gamma_k / var0
        weight = 1.0 - (k / (max_lag + 1.0))  # Bartlett kernel
        rho_sum += weight * rho_k

    adj = np.sqrt(max(1e-12, 1.0 + 2.0 * rho_sum))
    return float(base_sr / adj)


def _compute_ic_metrics(preds_df: pd.DataFrame, rank_rebalance_stride: int = 1) -> dict:
    """Compute daily cross-sectional Spearman IC diagnostics."""
    if "target_value" not in preds_df.columns:
        return {"ic_mean": None, "ic_std": None, "ic_ir": None, "ic_days": 0}

    eval_df = preds_df.dropna(subset=["target_value"]).copy()
    if rank_rebalance_stride > 1:
        rebalance_dates = sorted(eval_df["date"].dropna().unique())[::rank_rebalance_stride]
        eval_df = eval_df[eval_df["date"].isin(rebalance_dates)]

    ics: list[float] = []
    for _, day_df in eval_df.groupby("date"):
        if len(day_df) < 10:
            continue
        ic = day_df["probability_up"].corr(day_df["target_value"], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))

    if not ics:
        return {"ic_mean": None, "ic_std": None, "ic_ir": None, "ic_days": 0}

    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    periods_per_year = 252.0 / float(max(1, rank_rebalance_stride))
    ic_ir = float((ic_mean / ic_std) * np.sqrt(periods_per_year)) if ic_std > 0 else None
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_days": len(ics),
    }


def _compute_decile_factor_metrics(preds_df: pd.DataFrame, rank_rebalance_stride: int = 1) -> dict:
    """Compute top-decile minus bottom-decile factor return diagnostics."""
    if "target_value" not in preds_df.columns:
        return {
            "decile_spread_mean": None,
            "decile_spread_sharpe": None,
            "decile_spread_max_drawdown": None,
            "decile_spread_days": 0,
            "decile_curve": [],
        }

    eval_df = preds_df.dropna(subset=["target_value"]).copy()
    if rank_rebalance_stride > 1:
        rebalance_dates = sorted(eval_df["date"].dropna().unique())[::rank_rebalance_stride]
        eval_df = eval_df[eval_df["date"].isin(rebalance_dates)]

    daily = []
    for dt, day_df in eval_df.groupby("date"):
        day = day_df.sort_values("probability_up", ascending=False)
        n = len(day) // 10
        if n < 1:
            continue
        long_leg = float(day.head(n)["target_value"].mean())
        short_leg = float(day.tail(n)["target_value"].mean())
        daily.append({"date": dt, "decile_spread": long_leg - short_leg})

    if not daily:
        return {
            "decile_spread_mean": None,
            "decile_spread_sharpe": None,
            "decile_spread_max_drawdown": None,
            "decile_spread_days": 0,
            "decile_curve": [],
        }

    ddf = pd.DataFrame(daily).sort_values("date")
    ddf["cum"] = (1 + ddf["decile_spread"]).cumprod()
    rolling_peak = ddf["cum"].cummax()
    drawdown = (ddf["cum"] / rolling_peak) - 1

    mean_spread = float(ddf["decile_spread"].mean())
    std_spread = float(ddf["decile_spread"].std())
    periods_per_year = 252.0 / float(max(1, rank_rebalance_stride))
    sharpe = float((mean_spread / std_spread) * np.sqrt(periods_per_year)) if std_spread > 0 else None
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else None

    curve = [
        {"date": str(pd.Timestamp(row["date"]).date()), "cum_decile_spread": float(row["cum"])}
        for _, row in ddf.iterrows()
    ]
    return {
        "decile_spread_mean": mean_spread,
        "decile_spread_sharpe": sharpe,
        "decile_spread_max_drawdown": max_dd,
        "decile_spread_days": len(ddf),
        "decile_curve": curve,
    }


def _compute_rolling_ic_metrics(preds_df: pd.DataFrame, rank_rebalance_stride: int = 1, window: int = 60) -> dict:
    """Compute rolling IC mean and IR over a fixed date window."""
    if "target_value" not in preds_df.columns:
        return {
            "rolling_ic_window": window,
            "rolling_ic_curve": [],
            "rolling_ic_latest_mean": None,
            "rolling_ic_latest_ir": None,
        }

    eval_df = preds_df.dropna(subset=["target_value"]).copy()
    if rank_rebalance_stride > 1:
        rebalance_dates = sorted(eval_df["date"].dropna().unique())[::rank_rebalance_stride]
        eval_df = eval_df[eval_df["date"].isin(rebalance_dates)]

    by_date = []
    for dt, day_df in eval_df.groupby("date"):
        if len(day_df) < 10:
            continue
        ic = day_df["probability_up"].corr(day_df["target_value"], method="spearman")
        if pd.notna(ic):
            by_date.append({"date": pd.Timestamp(dt), "ic": float(ic)})

    if not by_date:
        return {
            "rolling_ic_window": window,
            "rolling_ic_curve": [],
            "rolling_ic_latest_mean": None,
            "rolling_ic_latest_ir": None,
        }

    ic_df = pd.DataFrame(by_date).sort_values("date")
    periods_per_year = 252.0 / float(max(1, rank_rebalance_stride))
    ic_df["rolling_ic_mean"] = ic_df["ic"].rolling(window).mean()
    ic_df["rolling_ic_std"] = ic_df["ic"].rolling(window).std()
    ic_df["rolling_ic_ir"] = (ic_df["rolling_ic_mean"] / ic_df["rolling_ic_std"]) * np.sqrt(periods_per_year)

    curve = [
        {
            "date": str(row["date"].date()),
            "ic": float(row["ic"]),
            "rolling_ic_mean": float(row["rolling_ic_mean"]) if pd.notna(row["rolling_ic_mean"]) else None,
            "rolling_ic_ir": float(row["rolling_ic_ir"]) if pd.notna(row["rolling_ic_ir"]) else None,
        }
        for _, row in ic_df.iterrows()
    ]
    latest = ic_df.iloc[-1]
    return {
        "rolling_ic_window": window,
        "rolling_ic_curve": curve,
        "rolling_ic_latest_mean": float(latest["rolling_ic_mean"]) if pd.notna(latest["rolling_ic_mean"]) else None,
        "rolling_ic_latest_ir": float(latest["rolling_ic_ir"]) if pd.notna(latest["rolling_ic_ir"]) else None,
    }


def _compute_regime_ic_metrics(preds_df: pd.DataFrame, rank_rebalance_stride: int = 1) -> dict:
    """Compute IC segmented by regime label and VIX bucket."""
    if "target_value" not in preds_df.columns:
        return {"regime_ic": {}, "vix_bucket_ic": {}}

    eval_df = preds_df.dropna(subset=["target_value"]).copy()
    if rank_rebalance_stride > 1:
        rebalance_dates = sorted(eval_df["date"].dropna().unique())[::rank_rebalance_stride]
        eval_df = eval_df[eval_df["date"].isin(rebalance_dates)]

    def _segment_ic(segment_df: pd.DataFrame, group_col: str) -> dict:
        out: dict[str, float | int | None] = {}
        for key, grp in segment_df.groupby(group_col):
            if key is None or (isinstance(key, float) and pd.isna(key)):
                continue
            daily_ics = []
            for _, day_df in grp.groupby("date"):
                if len(day_df) < 10:
                    continue
                ic = day_df["probability_up"].corr(day_df["target_value"], method="spearman")
                if pd.notna(ic):
                    daily_ics.append(float(ic))
            if daily_ics:
                out[str(key)] = {
                    "ic_mean": float(np.mean(daily_ics)),
                    "ic_std": float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else 0.0,
                    "n_days": len(daily_ics),
                }
        return out

    regime_ic = _segment_ic(eval_df[eval_df["regime_label"].notna()], "regime_label") if "regime_label" in eval_df.columns else {}

    vix_ic = {}
    if "vix_level" in eval_df.columns and eval_df["vix_level"].notna().any():
        tmp = eval_df[eval_df["vix_level"].notna()].copy()
        q1 = tmp["vix_level"].quantile(0.33)
        q2 = tmp["vix_level"].quantile(0.67)
        tmp["vix_bucket"] = np.where(
            tmp["vix_level"] <= q1,
            "low_vol",
            np.where(tmp["vix_level"] <= q2, "mid_vol", "high_vol"),
        )
        vix_ic = _segment_ic(tmp, "vix_bucket")

    return {
        "regime_ic": regime_ic,
        "vix_bucket_ic": vix_ic,
    }


def _compute_decile_monotonicity_metrics(preds_df: pd.DataFrame, rank_rebalance_stride: int = 1) -> dict:
    """Compute mean target return by prediction decile and monotonicity score."""
    if "target_value" not in preds_df.columns:
        return {"decile_return_table": {}, "decile_monotonicity_spearman": None}

    eval_df = preds_df.dropna(subset=["target_value"]).copy()
    if rank_rebalance_stride > 1:
        rebalance_dates = sorted(eval_df["date"].dropna().unique())[::rank_rebalance_stride]
        eval_df = eval_df[eval_df["date"].isin(rebalance_dates)]

    rows = []
    for _, day_df in eval_df.groupby("date"):
        day = day_df.sort_values("probability_up")
        n = len(day)
        if n < 20:
            continue
        # 1..10 deciles by prediction rank
        day = day.assign(
            decile=(np.floor(np.arange(n) * 10 / n).astype(int) + 1)
        )
        rows.append(day[["decile", "target_value"]])

    if not rows:
        return {"decile_return_table": {}, "decile_monotonicity_spearman": None}

    all_rows = pd.concat(rows, ignore_index=True)
    decile_mean = all_rows.groupby("decile")["target_value"].mean().to_dict()
    ordered_vals = [float(decile_mean.get(i, np.nan)) for i in range(1, 11)]
    valid = [(i, v) for i, v in enumerate(ordered_vals, start=1) if pd.notna(v)]
    if len(valid) >= 3:
        idx = pd.Series([x[0] for x in valid], dtype=float)
        vals = pd.Series([x[1] for x in valid], dtype=float)
        mono = idx.corr(vals, method="spearman")
        mono_val = float(mono) if pd.notna(mono) else None
    else:
        mono_val = None
    return {
        "decile_return_table": {str(k): float(v) for k, v in decile_mean.items()},
        "decile_monotonicity_spearman": mono_val,
    }


def _compute_probability_bin_metrics(preds_df: pd.DataFrame, rank_rebalance_stride: int = 1) -> dict:
    """Compute realized return by prediction-probability bins."""
    if "target_value" not in preds_df.columns:
        return {"probability_return_bins": []}

    eval_df = preds_df.dropna(subset=["target_value"]).copy()
    if rank_rebalance_stride > 1:
        rebalance_dates = sorted(eval_df["date"].dropna().unique())[::rank_rebalance_stride]
        eval_df = eval_df[eval_df["date"].isin(rebalance_dates)]

    bins = np.linspace(0.0, 1.0, 11)
    tmp = eval_df.copy()
    tmp["prob_bin"] = pd.cut(tmp["probability_up"], bins=bins, include_lowest=True)
    out = []
    for b, grp in tmp.groupby("prob_bin", observed=False):
        if len(grp) == 0:
            continue
        out.append({
            "bin": str(b),
            "count": int(len(grp)),
            "mean_probability": float(grp["probability_up"].mean()),
            "mean_realized_return": float(grp["target_value"].mean()),
        })
    return {"probability_return_bins": out}


def _compute_dispersion_metrics(preds_df: pd.DataFrame, rank_rebalance_stride: int = 1) -> dict:
    """Cross-sectional dispersion diagnostics and relation to IC/VIX."""
    if "target_value" not in preds_df.columns:
        return {
            "dispersion_mean": None,
            "dispersion_p25": None,
            "dispersion_p75": None,
            "ic_dispersion_corr": None,
            "ic_vix_corr": None,
            "ic_vix_dispersion_curve": [],
        }

    eval_df = preds_df.dropna(subset=["target_value"]).copy()
    if rank_rebalance_stride > 1:
        rebalance_dates = sorted(eval_df["date"].dropna().unique())[::rank_rebalance_stride]
        eval_df = eval_df[eval_df["date"].isin(rebalance_dates)]

    rows = []
    for dt, day_df in eval_df.groupby("date"):
        if len(day_df) < 10:
            continue
        ic = day_df["probability_up"].corr(day_df["target_value"], method="spearman")
        ret_col = "next_day_return" if "next_day_return" in day_df.columns else "target_value"
        dispersion = day_df[ret_col].std()
        vix = day_df["vix_level"].median() if "vix_level" in day_df.columns and day_df["vix_level"].notna().any() else np.nan
        rows.append({
            "date": pd.Timestamp(dt),
            "ic": float(ic) if pd.notna(ic) else np.nan,
            "dispersion": float(dispersion) if pd.notna(dispersion) else np.nan,
            "vix": float(vix) if pd.notna(vix) else np.nan,
        })

    if not rows:
        return {
            "dispersion_mean": None,
            "dispersion_p25": None,
            "dispersion_p75": None,
            "ic_dispersion_corr": None,
            "ic_vix_corr": None,
            "ic_vix_dispersion_curve": [],
        }

    ddf = pd.DataFrame(rows).sort_values("date")
    disp = ddf["dispersion"].dropna()
    ic_disp_corr = ddf["ic"].corr(ddf["dispersion"], method="spearman")
    ic_vix_corr = ddf["ic"].corr(ddf["vix"], method="spearman")
    curve = [
        {
            "date": str(row["date"].date()),
            "ic": float(row["ic"]) if pd.notna(row["ic"]) else None,
            "dispersion": float(row["dispersion"]) if pd.notna(row["dispersion"]) else None,
            "vix": float(row["vix"]) if pd.notna(row["vix"]) else None,
        }
        for _, row in ddf.iterrows()
    ]
    return {
        "dispersion_mean": float(disp.mean()) if len(disp) else None,
        "dispersion_p25": float(disp.quantile(0.25)) if len(disp) else None,
        "dispersion_p75": float(disp.quantile(0.75)) if len(disp) else None,
        "ic_dispersion_corr": float(ic_disp_corr) if pd.notna(ic_disp_corr) else None,
        "ic_vix_corr": float(ic_vix_corr) if pd.notna(ic_vix_corr) else None,
        "ic_vix_dispersion_curve": curve,
    }
