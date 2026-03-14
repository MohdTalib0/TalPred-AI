"""Live paper trading monitor — runs daily after market close.

What it does each day:
  1. Loads the production model + today's feature snapshots
  2. Generates predictions for all 503 symbols
  3. Ranks stocks: long top-20, short bottom-20
  4. Records the paper portfolio positions
  5. On subsequent days: computes realized return for yesterday's predictions
  6. Logs live IC, live Sharpe (rolling 21d), turnover, drawdown

Tracks:
  - live_ic:      Spearman(probability_up, next_5d_return) per day
  - rolling_ic21: rolling 21-day IC mean
  - live_sharpe:  rolling 21-day Sharpe of long-short portfolio
  - max_drawdown: running max drawdown since start

Usage:
  # Run once per day after market close:
  python -m scripts.paper_trading_monitor

  # Backfill recent days (e.g. to seed after promotion):
  python -m scripts.paper_trading_monitor --backfill-days 30

  # View current dashboard:
  python -m scripts.paper_trading_monitor --report-only
"""

import argparse
import logging
import os
from datetime import UTC, date, datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy import stats
from sqlalchemy import text

load_dotenv()

from src.db import SessionLocal
from src.models.schema import Symbol
from src.models.trainer import prepare_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("paper_monitor")

TOP_N = 20
HORIZON_DAYS = 5
ROLLING_WINDOW = 21       # days for rolling IC / Sharpe
IC_SUSPEND_THRESHOLD = 0.01   # IC < this for 10 days → suspend
IC_REDUCE_THRESHOLD  = 0.02   # IC < this for 10 days → reduce exposure


# ─── data helpers ──────────────────────────────────────────────────────────────

PROD_MODEL_DIR = os.path.join("artifacts", "production_model")


def _get_production_model():
    """Load the exact production model from local artifact.

    train_and_promote saves model.json + train_medians.pkl + metadata.json
    locally after every training run. This ensures paper trading uses the
    exact same model that was promoted — no retraining, no DagHub download.
    """
    model_path = os.path.join(PROD_MODEL_DIR, "model.json")
    medians_path = os.path.join(PROD_MODEL_DIR, "train_medians.pkl")
    metadata_path = os.path.join(PROD_MODEL_DIR, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No production model at {model_path}. "
            "Run train_and_promote first (it saves the artifact locally)."
        )

    import json
    import pickle
    import xgboost as xgb

    # Load metadata
    with open(metadata_path) as f:
        meta = json.load(f)
    logger.info(f"  Model version:    {meta.get('model_version')}")
    logger.info(f"  Feature profile:  {meta.get('feature_profile')}")
    logger.info(f"  Target mode:      {meta.get('target_mode')}")
    logger.info(f"  MLflow run:       {meta.get('mlflow_run_id')}")
    logger.info(f"  Promoted:         {meta.get('promoted')}")

    # Load XGBoost model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Load training medians (for NaN filling)
    with open(medians_path, "rb") as f:
        train_medians = pickle.load(f)

    logger.info(f"  Loaded production model from {PROD_MODEL_DIR}/")
    return model, train_medians, meta


def _load_features_for_date(session_date: date) -> pd.DataFrame:
    """Load feature snapshots for all active symbols on a given session date."""
    db = SessionLocal()
    try:
        # Use SELECT * to avoid hard-coding column names — handles schema evolution
        result = db.execute(text("""
            SELECT fs.*, s.sector
            FROM features_snapshot fs
            JOIN symbols s ON s.symbol = fs.symbol
            WHERE fs.target_session_date = :d
              AND s.is_active = true
        """), {"d": session_date})
        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()
        cols = list(result.keys())
        return pd.DataFrame(rows, columns=cols)
    finally:
        db.close()


def _load_realized_returns(session_date: date, horizon: int = 5) -> pd.DataFrame:
    """Load actual forward returns for predictions made on session_date.

    Uses simple (close[t+horizon] / close[t]) - 1.
    """
    end_date = session_date + timedelta(days=horizon + 5)  # extra buffer for weekends
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT symbol,
                   (end_close / start_close) - 1 AS fwd_return
            FROM (
                SELECT
                    symbol,
                    FIRST_VALUE(adj_close) OVER w AS start_close,
                    NTH_VALUE(adj_close, :h) OVER w AS end_close,
                    ROW_NUMBER() OVER w AS rn
                FROM market_bars_daily
                WHERE symbol IN (
                    SELECT DISTINCT symbol FROM features_snapshot
                    WHERE target_session_date = :sd
                )
                  AND date >= :sd AND date <= :ed
                WINDOW w AS (PARTITION BY symbol ORDER BY date
                             ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
            ) sub
            WHERE rn = 1 AND end_close IS NOT NULL AND start_close > 0
        """), {"sd": session_date, "ed": end_date, "h": horizon + 1}).fetchall()

        if not rows:
            return pd.DataFrame(columns=["symbol", "fwd_return"])
        return pd.DataFrame(rows, columns=["symbol", "fwd_return"])
    finally:
        db.close()


def _load_paper_log() -> pd.DataFrame:
    """Load existing paper trading log from artifacts."""
    path = "artifacts/paper_trading/daily_log.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["session_date"])
    return pd.DataFrame(columns=[
        "session_date", "n_symbols", "n_long", "n_short",
        "long_short_gross", "long_short_net", "ic_spearman",
        "rolling_ic21", "rolling_sharpe21", "cumulative_return",
        "max_drawdown", "avg_turnover", "deployment_status",
    ])


def _save_paper_log(df: pd.DataFrame):
    path = "artifacts/paper_trading/daily_log.csv"
    os.makedirs("artifacts/paper_trading", exist_ok=True)
    df.to_csv(path, index=False)


def _save_positions(session_date: date, long_syms: list, short_syms: list, probs: pd.Series):
    path = f"artifacts/paper_trading/positions_{session_date.isoformat()}.csv"
    rows = (
        [{"date": session_date, "symbol": s, "side": "long",  "prob_up": probs.get(s, 0.5)} for s in long_syms]
        + [{"date": session_date, "symbol": s, "side": "short", "prob_up": probs.get(s, 0.5)} for s in short_syms]
    )
    pd.DataFrame(rows).to_csv(path, index=False)


# ─── core daily step ───────────────────────────────────────────────────────────

def run_daily(session_date: date, model, train_medians, calibrator, log_df: pd.DataFrame) -> dict:
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Processing: {session_date}")

    # Load features
    feat_df = _load_features_for_date(session_date)
    if feat_df.empty:
        logger.warning(f"  No features found for {session_date} — skipping")
        return {}

    n_symbols = len(feat_df)
    logger.info(f"  Loaded features: {n_symbols} symbols")

    # Generate predictions
    # The model expects features that include cross-sectional transforms
    # (*_cs, *_sector_neutral) and liquidity features (log_market_cap, etc.)
    # which are computed dynamically by build_training_dataset, not stored in DB.
    # We must compute them here for inference.
    try:
        from src.features.leakage import _add_cross_sectional_transforms

        inf_df = feat_df.copy()

        # Compute liquidity features from market bar data on same date
        if "dollar_volume" not in inf_df.columns or inf_df["dollar_volume"].isna().all():
            _db = SessionLocal()
            try:
                bars = _db.execute(text("""
                    SELECT mb.symbol, mb.close, mb.volume,
                           s.market_cap
                    FROM market_bars_daily mb
                    JOIN symbols s ON s.symbol = mb.symbol
                    WHERE mb.date = :d AND s.is_active = true
                """), {"d": session_date}).fetchall()
                if bars:
                    bar_df = pd.DataFrame(bars, columns=["symbol", "close", "volume", "market_cap"])
                    bar_df["dollar_volume"] = bar_df["close"] * bar_df["volume"]
                    bar_df["log_market_cap"] = np.log(bar_df["market_cap"].clip(lower=1))
                    bar_df["turnover_ratio"] = bar_df["volume"] / (bar_df["market_cap"] / bar_df["close"]).clip(lower=1)
                    bar_df["market_cap_rank"] = bar_df["market_cap"].rank(pct=True)
                    bar_df["dollar_volume_rank_market"] = bar_df["dollar_volume"].rank(pct=True)
                    for col in ["dollar_volume", "log_market_cap", "turnover_ratio",
                                "market_cap_rank", "dollar_volume_rank_market"]:
                        if col not in inf_df.columns:
                            inf_df = inf_df.merge(bar_df[["symbol", col]], on="symbol", how="left")
                        else:
                            mask = inf_df[col].isna()
                            if mask.any():
                                fill = bar_df.set_index("symbol")[col]
                                inf_df.loc[mask, col] = inf_df.loc[mask, "symbol"].map(fill)
            finally:
                _db.close()

        # Compute turnover_acceleration from turnover_ratio
        if "turnover_acceleration" not in inf_df.columns or inf_df["turnover_acceleration"].isna().all():
            inf_df["turnover_acceleration"] = 0.0

        # Cross-sectional transforms: z-score and sector-neutral
        transform_cols = [
            "momentum_20d", "momentum_60d", "volume_change_5d",
            "volume_zscore_20d", "rolling_volatility_20d", "turnover_ratio",
        ]
        inf_df["target_session_date"] = session_date
        inf_df = _add_cross_sectional_transforms(inf_df, transform_cols)

        # Regime dummies
        if "regime_label" in inf_df.columns:
            regime_dummies = pd.get_dummies(inf_df["regime_label"], prefix="regime", dtype=float)
            inf_df = pd.concat([inf_df, regime_dummies], axis=1)

        # Use metadata to get the EXACT feature columns the model was trained on.
        # This handles regime dummies correctly (only include columns the model saw).
        import json as _json
        with open(os.path.join(PROD_MODEL_DIR, "metadata.json")) as _f:
            _meta = _json.load(_f)
        expected_cols = _meta["feature_columns"]

        for col in expected_cols:
            if col not in inf_df.columns:
                inf_df[col] = 0.0  # missing regime dummies default to 0

        X = inf_df[expected_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        # Fill NaN using training medians (only for columns present in both)
        for col in expected_cols:
            if col in train_medians.index and X[col].isna().any():
                X[col] = X[col].fillna(train_medians[col])

        feat_df["prob_up"] = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.error(f"  Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

    # Rank and build long/short portfolio
    feat_df = feat_df.sort_values("prob_up", ascending=False)
    long_picks  = feat_df.head(TOP_N)["symbol"].tolist()
    short_picks = feat_df.tail(TOP_N)["symbol"].tolist()
    prob_map = dict(zip(feat_df["symbol"], feat_df["prob_up"]))

    _save_positions(session_date, long_picks, short_picks, prob_map)

    # Compute realized return for predictions made HORIZON_DAYS ago
    past_date = session_date - timedelta(days=HORIZON_DAYS + 3)  # rough offset
    ic_val = np.nan
    long_short_gross = np.nan
    long_short_net = np.nan

    past_pos_path = f"artifacts/paper_trading/positions_{past_date.isoformat()}.csv"
    for delta in range(0, 8):
        check_date = past_date + timedelta(days=delta)
        candidate = f"artifacts/paper_trading/positions_{check_date.isoformat()}.csv"
        if os.path.exists(candidate):
            past_pos_path = candidate
            past_date = check_date
            break

    if os.path.exists(past_pos_path):
        past_pos = pd.read_csv(past_pos_path)
        realized = _load_realized_returns(past_date, horizon=HORIZON_DAYS)
        if not realized.empty and "fwd_return" in realized.columns:
            merged = past_pos.merge(realized, on="symbol", how="inner").dropna(subset=["fwd_return"])
            if len(merged) >= 5:
                # IC: rank correlation of prob_up vs fwd_return for all symbols
                all_syms_realized = feat_df[["symbol", "prob_up"]].merge(realized, on="symbol", how="inner")
                if len(all_syms_realized) >= 10:
                    ic_val = float(stats.spearmanr(
                        all_syms_realized["prob_up"],
                        all_syms_realized["fwd_return"]
                    ).correlation)

                long_ret  = merged[merged["side"] == "long"]["fwd_return"].mean()
                short_ret = merged[merged["side"] == "short"]["fwd_return"].mean()
                long_short_gross = float(long_ret - short_ret) if pd.notna(long_ret) and pd.notna(short_ret) else np.nan
                cost_per_trade = 0.001  # 10 bps each leg
                long_short_net = long_short_gross - 2 * cost_per_trade if pd.notna(long_short_gross) else np.nan

    # Compute turnover vs yesterday
    avg_turnover = np.nan
    yesterday = session_date - timedelta(days=1)
    for delta in range(1, 5):
        yd = session_date - timedelta(days=delta)
        yd_path = f"artifacts/paper_trading/positions_{yd.isoformat()}.csv"
        if os.path.exists(yd_path):
            yd_pos = pd.read_csv(yd_path)
            yd_long  = set(yd_pos[yd_pos["side"] == "long"]["symbol"])
            yd_short = set(yd_pos[yd_pos["side"] == "short"]["symbol"])
            cur_long  = set(long_picks)
            cur_short = set(short_picks)
            long_changed  = len(yd_long.symmetric_difference(cur_long))
            short_changed = len(yd_short.symmetric_difference(cur_short))
            avg_turnover = (long_changed + short_changed) / (2 * TOP_N)
            break

    # Rolling IC and Sharpe
    cutoff = pd.Timestamp(session_date - timedelta(days=ROLLING_WINDOW + 5))
    log_dates = pd.to_datetime(log_df["session_date"])
    recent = log_df[log_dates >= cutoff]
    ic_series = pd.to_numeric(recent["ic_spearman"], errors="coerce").dropna()
    ls_series = pd.to_numeric(recent["long_short_net"], errors="coerce").dropna()

    if not np.isnan(ic_val):
        ic_series = pd.concat([ic_series, pd.Series([ic_val])], ignore_index=True)
    rolling_ic21 = float(ic_series.tail(ROLLING_WINDOW).mean()) if len(ic_series) >= 3 else np.nan

    if not np.isnan(long_short_net) if not (isinstance(long_short_net, float) and np.isnan(long_short_net)) else False:
        ls_series = pd.concat([ls_series, pd.Series([long_short_net])], ignore_index=True)
    ls_tail = ls_series.tail(ROLLING_WINDOW)
    # Portfolio is rebalanced daily → annualize with sqrt(252), not sqrt(252/horizon)
    rolling_sharpe21 = float(ls_tail.mean() / ls_tail.std() * np.sqrt(252)) if len(ls_tail) >= 5 and ls_tail.std() > 0 else np.nan

    # Cumulative return and drawdown
    all_ls = pd.to_numeric(log_df["long_short_net"], errors="coerce").dropna()
    if not np.isnan(long_short_net):
        all_ls = pd.concat([all_ls, pd.Series([long_short_net])], ignore_index=True)
    cum_ret = float((1 + all_ls).prod() - 1) if len(all_ls) > 0 else 0.0
    cum_curve = (1 + all_ls).cumprod()
    running_max = cum_curve.cummax()
    dd = (cum_curve / running_max - 1)
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # Deployment guard — only apply after minimum sample to avoid early noise triggers
    MIN_IC_SAMPLES = 30
    deployment_status = "ACTIVE"
    n_ic_samples = len(ic_series)
    if n_ic_samples < MIN_IC_SAMPLES:
        deployment_status = f"WARMUP ({n_ic_samples}/{MIN_IC_SAMPLES} days)"
    elif rolling_ic21 is not None and not np.isnan(rolling_ic21):
        if rolling_ic21 < IC_SUSPEND_THRESHOLD:
            deployment_status = "SUSPENDED (IC < 0.01)"
        elif rolling_ic21 < IC_REDUCE_THRESHOLD:
            deployment_status = "REDUCED (IC < 0.02)"

    row = {
        "session_date":     session_date,
        "n_symbols":        n_symbols,
        "n_long":           len(long_picks),
        "n_short":          len(short_picks),
        "long_short_gross": round(long_short_gross, 6) if not np.isnan(long_short_gross) else None,
        "long_short_net":   round(long_short_net, 6)   if not np.isnan(long_short_net) else None,
        "ic_spearman":      round(ic_val, 4)            if not np.isnan(ic_val) else None,
        "rolling_ic21":     round(rolling_ic21, 4)      if rolling_ic21 and not np.isnan(rolling_ic21) else None,
        "rolling_sharpe21": round(rolling_sharpe21, 3)  if rolling_sharpe21 and not np.isnan(rolling_sharpe21) else None,
        "cumulative_return": round(cum_ret, 4),
        "max_drawdown":     round(max_dd, 4),
        "avg_turnover":     round(avg_turnover, 3)      if avg_turnover and not np.isnan(avg_turnover) else None,
        "deployment_status": deployment_status,
    }

    logger.info(f"  Positions: long={len(long_picks)} | short={len(short_picks)}")
    if not np.isnan(ic_val):
        logger.info(f"  Live IC (spearman):     {ic_val:+.4f}")
    if rolling_ic21 and not np.isnan(rolling_ic21):
        logger.info(f"  Rolling IC (21d):       {rolling_ic21:+.4f}")
    if rolling_sharpe21 and not np.isnan(rolling_sharpe21):
        logger.info(f"  Rolling Sharpe (21d):   {rolling_sharpe21:.3f}")
    if not np.isnan(long_short_net):
        logger.info(f"  Long-short net return:  {long_short_net*100:+.2f}%")
    logger.info(f"  Cumulative return:      {cum_ret*100:+.2f}%")
    logger.info(f"  Max drawdown:           {max_dd*100:.2f}%")
    logger.info(f"  Deployment guard:       {deployment_status}")

    return row


# ─── report ────────────────────────────────────────────────────────────────────

def print_report(log_df: pd.DataFrame):
    if log_df.empty:
        logger.info("No paper trading data yet.")
        return

    logger.info("\n" + "=" * 65)
    logger.info("  PAPER TRADING DASHBOARD")
    logger.info("=" * 65)
    logger.info(f"  Days tracked:    {len(log_df)}")
    min_dt = log_df["session_date"].min()
    max_dt = log_df["session_date"].max()
    logger.info(f"  From:            {min_dt.date() if hasattr(min_dt, 'date') else min_dt}")
    logger.info(f"  To:              {max_dt.date() if hasattr(max_dt, 'date') else max_dt}")

    ls_net = pd.to_numeric(log_df["long_short_net"], errors="coerce").dropna()
    ic_vals = pd.to_numeric(log_df["ic_spearman"], errors="coerce").dropna()

    if len(ls_net) > 0:
        annualised_ret = float((1 + ls_net.mean()) ** 252 - 1)
        sharpe = float(ls_net.mean() / ls_net.std() * np.sqrt(252)) if ls_net.std() > 0 else 0
        logger.info(f"\n  RETURNS (realized, {len(ls_net)} periods):")
        logger.info(f"    Mean net return/period:  {ls_net.mean()*100:+.2f}%")
        logger.info(f"    Annualised return:        {annualised_ret*100:+.1f}%")
        logger.info(f"    Live Sharpe:              {sharpe:.2f}")
        logger.info(f"    Win rate:                 {(ls_net > 0).mean()*100:.1f}%")
        logger.info(f"    Cumulative:               {log_df['cumulative_return'].iloc[-1]*100:+.2f}%")
        logger.info(f"    Max drawdown:             {log_df['max_drawdown'].min()*100:.2f}%")

    if len(ic_vals) > 0:
        logger.info(f"\n  SIGNAL QUALITY:")
        logger.info(f"    Mean live IC:             {ic_vals.mean():+.4f}")
        logger.info(f"    IC > 0 rate:              {(ic_vals > 0).mean()*100:.1f}%")
        latest_ic21 = pd.to_numeric(log_df["rolling_ic21"], errors="coerce").dropna()
        if len(latest_ic21) > 0:
            logger.info(f"    Latest rolling IC (21d):  {latest_ic21.iloc[-1]:+.4f}")

    latest = log_df.iloc[-1]
    logger.info(f"\n  LATEST STATUS: {latest.get('deployment_status', 'N/A')}")
    logger.info(f"  Target: live IC > 0.02 sustained for 30+ days = SIGNAL CONFIRMED")
    logger.info("=" * 65)


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper trading monitor")
    parser.add_argument("--backfill-days", type=int, default=0,
                        help="Process last N days retroactively (0 = today only)")
    parser.add_argument("--report-only", action="store_true",
                        help="Print dashboard without generating new predictions")
    args = parser.parse_args()

    os.makedirs("artifacts/paper_trading", exist_ok=True)
    log_df = _load_paper_log()

    if args.report_only:
        print_report(log_df)
        return

    # Load exact production model from local artifact (saved by train_and_promote)
    logger.info("Loading production model from local artifact...")
    try:
        model, train_medians, meta = _get_production_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Determine dates to process
    today = date.today()
    if args.backfill_days > 0:
        dates_to_process = [
            today - timedelta(days=i)
            for i in range(args.backfill_days, -1, -1)
        ]
    else:
        dates_to_process = [today]

    already_done = set(pd.to_datetime(log_df["session_date"]).dt.date) if not log_df.empty else set()
    new_rows = []

    for d in dates_to_process:
        if d in already_done:
            logger.info(f"  {d} already logged — skipping")
            continue
        row = run_daily(d, model, train_medians, None, log_df)
        if row:
            new_rows.append(row)
            # Append to log for rolling calculations in subsequent iterations
            log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)

    if new_rows:
        _save_paper_log(log_df)
        logger.info(f"\nSaved {len(new_rows)} new entries to artifacts/paper_trading/daily_log.csv")

    print_report(log_df)


if __name__ == "__main__":
    main()
