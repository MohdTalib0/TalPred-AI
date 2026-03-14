"""Paper trading simulation engine (BE-401).

Confidence-weighted allocation with position caps per ENG-SPEC 13.

Portfolio policy:
  weight_i = confidence_i / sum(confidence_j)
  max_position = 5%
  min_confidence_trade = 0.60
  daily rebalance, configurable transaction costs + slippage
"""

import logging
import uuid
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.schema import PaperTrade, SimulationRun

logger = logging.getLogger(__name__)


def run_simulation(
    db: Session,
    start_date: date,
    end_date: date,
    starting_capital: float = 100_000.0,
    min_confidence_trade: float = 0.60,
    max_position: float = 0.05,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    top_n: int = 20,
    model_version: str | None = None,
    enable_regime_guard: bool = True,
    vix_quarter_exposure_below: float = 12.0,
    vix_half_exposure_below: float = 15.0,
    vix_full_exposure_above: float = 20.0,
    use_live_ic_guard: bool = False,
    live_ic_lookback_days: int = 60,
    live_ic_floor: float = 0.01,
    low_ic_exposure_scale: float = 0.5,
) -> dict:
    """Execute a full paper-trading simulation over a date range.

    Returns dict with run_id, equity curve, and performance metrics.
    """
    run_id = uuid.uuid4().hex[:16]

    sim_run = SimulationRun(
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        starting_capital=starting_capital,
        min_confidence_trade=min_confidence_trade,
        max_position=max_position,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        model_version=model_version,
        status="running",
    )
    db.add(sim_run)
    db.commit()

    predictions_df = _load_predictions(db, start_date, end_date, model_version)
    if predictions_df.empty:
        sim_run.status = "failed"
        sim_run.result_metrics = {"error": "No predictions found"}
        db.commit()
        return {"run_id": run_id, "error": "No predictions found"}

    prices_df = _load_prices(db, start_date, end_date)
    if prices_df.empty:
        sim_run.status = "failed"
        sim_run.result_metrics = {"error": "No price data found"}
        db.commit()
        return {"run_id": run_id, "error": "No price data found"}

    trading_dates = sorted(predictions_df["target_date"].unique())
    logger.info(f"Simulation {run_id}: {len(trading_dates)} trading days, capital=${starting_capital:,.0f}")

    equity = starting_capital
    equity_curve = [{"date": str(start_date), "equity": equity}]
    all_trades = []
    daily_returns = []
    daily_exposure_scales = []
    live_ic_trace = []

    cost_factor = (transaction_cost_bps + slippage_bps) / 10_000

    for td in trading_dates:
        day_preds = predictions_df[predictions_df["target_date"] == td].copy()

        day_preds = day_preds[day_preds["confidence"] >= min_confidence_trade]
        if day_preds.empty:
            equity_curve.append({"date": str(td), "equity": equity})
            daily_returns.append(0.0)
            continue

        day_preds = day_preds.nlargest(top_n, "confidence")

        exposure_scale, live_ic = _compute_exposure_scale(
            db=db,
            day_preds=day_preds,
            as_of_date=td,
            model_version=model_version,
            enable_regime_guard=enable_regime_guard,
            vix_quarter_exposure_below=vix_quarter_exposure_below,
            vix_half_exposure_below=vix_half_exposure_below,
            vix_full_exposure_above=vix_full_exposure_above,
            use_live_ic_guard=use_live_ic_guard,
            live_ic_lookback_days=live_ic_lookback_days,
            live_ic_floor=live_ic_floor,
            low_ic_exposure_scale=low_ic_exposure_scale,
        )
        weights = _compute_weights(day_preds, max_position, exposure_scale=exposure_scale)
        daily_exposure_scales.append(float(exposure_scale))
        if live_ic is not None:
            live_ic_trace.append(float(live_ic))

        day_pnl = 0.0
        day_trades = []

        for _, pred in day_preds.iterrows():
            symbol = pred["symbol"]
            weight = weights.get(symbol, 0)
            if weight == 0:
                continue

            price_row = prices_df[
                (prices_df["symbol"] == symbol) & (prices_df["date"] == td)
            ]
            if price_row.empty:
                continue

            entry_price = float(price_row.iloc[0]["open"])
            exit_price = float(price_row.iloc[0]["close"])

            if entry_price <= 0:
                continue

            position_value = equity * weight
            direction_sign = 1.0 if pred["direction"] == "up" else -1.0
            shares = position_value / entry_price

            gross_return = direction_sign * (exit_price - entry_price) / entry_price
            trade_cost = position_value * cost_factor * 2
            net_pnl = position_value * gross_return - trade_cost

            day_pnl += net_pnl

            trade = {
                "run_id": run_id,
                "date": td,
                "symbol": symbol,
                "weight": weight,
                "position_qty": shares * direction_sign,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "transaction_cost": trade_cost,
                "slippage_cost": position_value * slippage_bps / 10_000,
                "daily_pnl": net_pnl,
            }
            day_trades.append(trade)

        equity += day_pnl
        daily_ret = day_pnl / (equity - day_pnl) if (equity - day_pnl) > 0 else 0
        daily_returns.append(daily_ret)
        equity_curve.append({"date": str(td), "equity": round(equity, 2)})
        all_trades.extend(day_trades)

    metrics = _compute_metrics(
        equity_curve, daily_returns, all_trades, starting_capital
    )
    metrics["avg_exposure_scale"] = round(float(np.mean(daily_exposure_scales)), 4) if daily_exposure_scales else 1.0
    metrics["min_exposure_scale"] = round(float(np.min(daily_exposure_scales)), 4) if daily_exposure_scales else 1.0
    metrics["use_live_ic_guard"] = bool(use_live_ic_guard)
    metrics["live_ic_avg"] = round(float(np.mean(live_ic_trace)), 5) if live_ic_trace else None

    _save_trades(db, all_trades)

    sim_run.status = "completed"
    sim_run.result_metrics = metrics
    db.commit()

    logger.info(
        f"Simulation {run_id} complete: "
        f"return={metrics['total_return_pct']:.2f}%, "
        f"sharpe={metrics['sharpe_ratio']:.3f}, "
        f"max_dd={metrics['max_drawdown_pct']:.2f}%"
    )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "n_trades": len(all_trades),
        "n_trading_days": len(trading_dates),
    }


def _load_predictions(
    db: Session, start_date: date, end_date: date, model_version: str | None
) -> pd.DataFrame:
    mv_filter = "AND p.model_version = :mv" if model_version else ""
    params = {"start": start_date, "end": end_date}
    if model_version:
        params["mv"] = model_version

    result = db.execute(text(f"""
        SELECT p.symbol, p.target_date, p.direction, p.probability_up, p.confidence, p.model_version,
               fs.vix_level, p.realized_return
        FROM predictions p
        LEFT JOIN features_snapshot fs ON p.feature_snapshot_id = fs.snapshot_id
        WHERE target_date >= :start AND target_date <= :end
        {mv_filter}
        ORDER BY target_date, confidence DESC
    """), params)

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "symbol", "target_date", "direction", "probability_up", "confidence", "model_version", "vix_level", "realized_return"
    ])
    return df


def _load_prices(db: Session, start_date: date, end_date: date) -> pd.DataFrame:
    result = db.execute(text("""
        SELECT symbol, date, open, close
        FROM market_bars_daily
        WHERE date >= :start AND date <= :end
        ORDER BY date
    """), {"start": start_date, "end": end_date})

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=["symbol", "date", "open", "close"])


def _compute_weights(preds_df: pd.DataFrame, max_position: float, exposure_scale: float = 1.0) -> dict[str, float]:
    """Confidence-weighted allocation with position cap."""
    total_confidence = preds_df["confidence"].sum()
    if total_confidence == 0:
        return {}

    weights = {}
    for _, row in preds_df.iterrows():
        raw_weight = row["confidence"] / total_confidence
        weights[row["symbol"]] = min(raw_weight, max_position)

    weight_sum = sum(weights.values())
    if weight_sum > 1.0:
        scale = 1.0 / weight_sum
        weights = {k: v * scale for k, v in weights.items()}

    exposure_scale = float(min(max(exposure_scale, 0.0), 1.0))
    if exposure_scale < 1.0:
        weights = {k: v * exposure_scale for k, v in weights.items()}

    return weights


def _compute_exposure_scale(
    db: Session,
    day_preds: pd.DataFrame,
    as_of_date: date,
    model_version: str | None,
    enable_regime_guard: bool,
    vix_quarter_exposure_below: float,
    vix_half_exposure_below: float,
    vix_full_exposure_above: float,
    use_live_ic_guard: bool,
    live_ic_lookback_days: int,
    live_ic_floor: float,
    low_ic_exposure_scale: float,
) -> tuple[float, float | None]:
    """Combine VIX-based and live-IC-based exposure controls."""
    if not enable_regime_guard:
        return 1.0, None

    vix_scale = 1.0
    vix = None
    if "vix_level" in day_preds.columns and day_preds["vix_level"].notna().any():
        vix = float(day_preds["vix_level"].median())
        if vix < vix_quarter_exposure_below:
            vix_scale = 0.25
        elif vix < vix_half_exposure_below:
            vix_scale = 0.50
        elif vix >= vix_full_exposure_above:
            vix_scale = 1.0
        else:
            vix_scale = 0.75

    live_ic = None
    ic_scale = 1.0
    if use_live_ic_guard:
        live_ic = _compute_live_rolling_ic(db, as_of_date, live_ic_lookback_days, model_version=model_version)
        if live_ic is not None and live_ic < live_ic_floor:
            ic_scale = float(min(max(low_ic_exposure_scale, 0.0), 1.0))

    return min(vix_scale, ic_scale), live_ic


def _compute_live_rolling_ic(
    db: Session,
    as_of_date: date,
    lookback_days: int,
    model_version: str | None = None,
) -> float | None:
    """Compute trailing mean daily IC from realized predictions before as_of_date."""
    mv_filter = "AND model_version = :mv" if model_version else ""
    params = {
        "start": as_of_date - pd.Timedelta(days=lookback_days),
        "end": as_of_date,
    }
    if model_version:
        params["mv"] = model_version

    result = db.execute(text(f"""
        SELECT target_date, probability_up, realized_return
        FROM predictions
        WHERE target_date >= :start
          AND target_date < :end
          AND realized_return IS NOT NULL
          {mv_filter}
        ORDER BY target_date
    """), params)
    rows = result.fetchall()
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["target_date", "probability_up", "realized_return"])
    if df.empty:
        return None
    daily_ics = []
    for _, day_df in df.groupby("target_date"):
        if len(day_df) < 10:
            continue
        ic = day_df["probability_up"].corr(day_df["realized_return"], method="spearman")
        if pd.notna(ic):
            daily_ics.append(float(ic))
    if not daily_ics:
        return None
    return float(np.mean(daily_ics))


def _compute_metrics(
    equity_curve: list[dict],
    daily_returns: list[float],
    trades: list[dict],
    starting_capital: float,
) -> dict:
    """Compute simulation performance metrics."""
    final_equity = equity_curve[-1]["equity"] if equity_curve else starting_capital
    total_return = (final_equity - starting_capital) / starting_capital

    returns = np.array(daily_returns) if daily_returns else np.array([0.0])
    trading_days = len(returns)

    sharpe = 0.0
    if returns.std() > 0 and trading_days > 1:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    wins = sum(1 for t in trades if t["daily_pnl"] > 0)
    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    total_costs = sum(t["transaction_cost"] for t in trades)

    unique_days = len(set(t["date"] for t in trades)) if trades else 0
    avg_daily_turnover = total_trades / unique_days if unique_days > 0 else 0

    return {
        "total_return_pct": round(total_return * 100, 4),
        "annualized_return_pct": round(total_return * (252 / max(trading_days, 1)) * 100, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "win_rate": round(win_rate, 4),
        "total_trades": total_trades,
        "total_trading_days": trading_days,
        "total_transaction_costs": round(total_costs, 2),
        "avg_daily_turnover": round(avg_daily_turnover, 2),
        "final_equity": round(final_equity, 2),
        "net_profit": round(final_equity - starting_capital, 2),
    }


def _save_trades(db: Session, trades: list[dict]) -> int:
    if not trades:
        return 0

    for t in trades:
        db.add(PaperTrade(**t))

    db.commit()
    return len(trades)
