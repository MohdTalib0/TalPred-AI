"""Paper trading simulation engine (BE-401).

Supports two modes:
  1. **Legacy** (backward-compatible): confidence-weighted allocation with
     position caps, VIX/IC exposure guards.
  2. **Strategy framework**: pluggable portfolio construction modes
     (momentum, sector-rotation, mean-reversion) with shared risk
     management, realistic cost modeling, benchmark tracking, and
     multi-day hold support.

Portfolio policy defaults (legacy mode):
  weight_i = confidence_i / sum(confidence_j)
  max_position = 5%
  min_confidence_trade = 0.60
  daily rebalance, configurable transaction costs + slippage
"""

import logging
import uuid
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.schema import PaperTrade, SimulationRun
from src.strategies.base import BaseStrategy
from src.strategies.config import StrategyFrameworkConfig
from src.strategies.portfolio import PortfolioConstructor
from src.strategies.risk_manager import RiskManager

logger = logging.getLogger(__name__)


def _try_build_factor_model(db: Session, end_date: date):
    """Attempt to build and fit a StatisticalFactorModel.

    Returns the fitted model or None if data is insufficient.
    """
    try:
        from src.models.factor_model import StatisticalFactorModel, build_return_matrix

        start = end_date - timedelta(days=400)
        symbols_result = db.execute(text(
            "SELECT DISTINCT symbol FROM market_bars_daily "
            "WHERE date >= :s AND date <= :e",
        ), {"s": start, "e": end_date})
        symbols = [r[0] for r in symbols_result.fetchall()]
        if len(symbols) < 40:
            logger.info("Factor model skipped: only %d symbols (need 40+)", len(symbols))
            return None

        returns = build_return_matrix(db, symbols, start, end_date)
        if returns.empty or len(returns) < 60:
            logger.info("Factor model skipped: insufficient return history")
            return None

        fm = StatisticalFactorModel(n_factors=30, lookback=252)
        fm.fit(returns, as_of_date=end_date)
        return fm
    except Exception as e:
        logger.warning("Factor model build failed: %s", e)
        return None


# -----------------------------------------------------------------------
# Strategy-framework entry point
# -----------------------------------------------------------------------


def run_strategy_simulation(
    db: Session,
    strategy: BaseStrategy,
    start_date: date,
    end_date: date,
    config: StrategyFrameworkConfig | None = None,
    model_version: str | None = None,
) -> dict:
    """Execute a paper-trading simulation using the strategy framework.

    Models true multi-day holds: positions are opened on rebalance days
    and held until the next rebalance.  Transaction costs are only charged
    on turnover (the delta between old and new weights), not every day.

    P&L on hold days = close-to-close return of existing positions.
    P&L on rebalance days = close-to-close return minus rebalance costs.
    """
    cfg = config or StrategyFrameworkConfig()
    run_id = uuid.uuid4().hex[:16]
    risk_mgr = RiskManager(cfg.risk)

    # Factor model is rebuilt rolling inside the simulation loop (quarterly)
    # to avoid lookahead bias — see rebalance block below.
    _factor_rebuild_every = 63  # ~quarterly in trading days
    _days_since_factor_build = _factor_rebuild_every  # force first build

    portfolio = PortfolioConstructor(
        cfg.cost, beta_neutral=cfg.beta_neutral,
        turnover_penalty=cfg.turnover_penalty,
    )

    sim_run = SimulationRun(
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        starting_capital=cfg.starting_capital,
        min_confidence_trade=getattr(
            getattr(cfg, strategy.name, cfg.momentum), "min_confidence", 0.60
        ),
        max_position=cfg.risk.max_position_weight,
        transaction_cost_bps=cfg.cost.base_cost_bps,
        slippage_bps=cfg.cost.slippage_floor_bps,
        model_version=model_version,
        strategy_name=strategy.name,
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

    features_df = _load_features(db, start_date, end_date)
    prices_df = _load_prices(db, start_date, end_date)
    market_df = _load_market_window(db, end_date, lookback=60)

    if prices_df.empty:
        sim_run.status = "failed"
        sim_run.result_metrics = {"error": "No price data found"}
        db.commit()
        return {"run_id": run_id, "error": "No price data found"}

    trading_dates = sorted(predictions_df["target_date"].unique())
    rebalance_dates = set(trading_dates[:: cfg.rebalance.stride_days])

    logger.info(
        f"Simulation {run_id} [{strategy.name}]: {len(trading_dates)} trading days, "
        f"{len(rebalance_dates)} rebalance days, capital=${cfg.starting_capital:,.0f}"
    )

    equity = cfg.starting_capital
    equity_curve = [{"date": str(start_date), "equity": equity}]
    all_trades: list[dict] = []
    daily_returns: list[float] = []
    benchmark_returns: list[float] = []
    daily_turnovers: list[float] = []

    # Multi-day hold state
    current_weights: dict[str, float] = {}
    position_values: dict[str, float] = {}   # actual dollar value (signed)
    position_pnl: dict[str, float] = {}      # accumulated P&L per position
    realized_position_pnls: list[float] = []
    prev_closes: dict[str, float] = {}

    for td in trading_dates:
        _days_since_factor_build += 1
        day_preds = predictions_df[predictions_df["target_date"] == td].copy()
        day_features = (
            features_df[features_df["target_session_date"] == td]
            if "target_session_date" in features_df.columns
            else features_df
        )

        is_rebalance = td in rebalance_dates

        # ── Close-to-close P&L for existing positions ──
        hold_pnl = 0.0
        for symbol in list(position_values.keys()):
            price_row = prices_df[
                (prices_df["symbol"] == symbol) & (prices_df["date"] == td)
            ]
            if price_row.empty:
                continue
            today_close = float(price_row.iloc[0]["close"])
            yesterday_close = prev_closes.get(symbol)
            if yesterday_close and yesterday_close > 0:
                ret = (today_close - yesterday_close) / yesterday_close
                pnl = position_values[symbol] * ret
                hold_pnl += pnl
                position_values[symbol] += pnl
                position_pnl[symbol] = position_pnl.get(symbol, 0.0) + pnl

        # ── Rebalance: compute new weights and turnover cost ──
        rebalance_cost = 0.0
        if is_rebalance:
            # Rebuild factor model rolling to avoid lookahead bias
            if (
                cfg.risk.factor_constraint_enabled
                and _days_since_factor_build >= _factor_rebuild_every
            ):
                factor_model = _try_build_factor_model(db, td)
                if factor_model is not None:
                    risk_mgr.set_factor_model(factor_model)
                    logger.info("Factor model rebuilt as of %s", td)
                else:
                    # Explicitly clear the old model so stale factor constraints
                    # are not silently applied. set_factor_model(None) also sets
                    # factor_constraint_enabled = False.
                    risk_mgr.set_factor_model(None)
                    logger.warning(
                        "Factor model rebuild failed as of %s — "
                        "factor constraints disabled until next successful build",
                        td,
                    )
                _days_since_factor_build = 0

            vix = _get_vix(day_features)
            signals = strategy.generate_signals(
                day_preds, day_features, market_df, td
            )
            signals = risk_mgr.apply(
                signals, day_features, market_df, equity, daily_returns, vix=vix
            )

            new_weights = portfolio.compute_target_weights(
                signals, market_df, prices_df, td,
                current_weights=current_weights,
            )

            adv_lookup = portfolio._adv_lookup(market_df)

            new_weights = portfolio.apply_partial_fills(
                new_weights, equity, adv_lookup
            )

            turnover = _compute_turnover(current_weights, new_weights)
            daily_turnovers.append(turnover)
            rebalance_cost = portfolio.compute_rebalance_cost(
                current_weights, new_weights, equity, adv_lookup
            )

            # Realize P&L for positions being closed or resized
            for symbol in current_weights:
                new_w = new_weights.get(symbol, 0.0)
                if abs(new_w) < 1e-8 or abs(new_w - current_weights[symbol]) > 1e-8:
                    if symbol in position_pnl:
                        realized_position_pnls.append(position_pnl.pop(symbol))

            # Record trades for positions that changed
            for symbol in set(list(current_weights.keys()) + list(new_weights.keys())):
                old_w = current_weights.get(symbol, 0.0)
                new_w = new_weights.get(symbol, 0.0)
                delta = new_w - old_w
                if abs(delta) < 1e-8:
                    continue
                price_row = prices_df[
                    (prices_df["symbol"] == symbol) & (prices_df["date"] == td)
                ]
                if price_row.empty:
                    continue
                close_p = float(price_row.iloc[0]["close"])
                position_value = equity * abs(delta)
                adv = adv_lookup.get(symbol, 0)
                slip = portfolio.cost.slippage_bps(position_value, adv)
                tx = position_value * (portfolio.cost.base_cost_bps / 10_000)
                sl = position_value * (slip / 10_000)
                borrow = 0.0
                if new_w < 0:
                    borrow = portfolio.cost.daily_borrow_cost(equity * abs(new_w))

                all_trades.append({
                    "run_id": run_id,
                    "date": td,
                    "symbol": symbol,
                    "weight": float(new_w),
                    "position_qty": float(equity * abs(new_w) / close_p) * (1 if new_w > 0 else -1) if close_p > 0 else 0.0,
                    "entry_price": float(close_p),
                    "exit_price": float(close_p),
                    "transaction_cost": float(tx + sl),
                    "slippage_cost": float(borrow),
                    "daily_pnl": 0.0,
                })

            current_weights = new_weights
            position_values = {
                sym: w * equity
                for sym, w in new_weights.items()
                if abs(w) > 1e-8
            }

        # Daily borrow cost for shorts (accrued daily, not just on rebalance)
        daily_borrow = sum(
            portfolio.cost.daily_borrow_cost(abs(v))
            for sym, v in position_values.items()
            if v < 0
        )

        day_pnl = hold_pnl - rebalance_cost - daily_borrow
        equity += day_pnl
        daily_ret = day_pnl / (equity - day_pnl) if (equity - day_pnl) > 0 else 0.0
        daily_returns.append(daily_ret)

        bench_ret = portfolio._benchmark_return(prices_df, td)
        benchmark_returns.append(bench_ret)
        equity_curve.append({"date": str(td), "equity": round(equity, 2)})

        # Update prev_closes for next day's close-to-close calculation
        for symbol in set(list(position_values.keys()) + list(prev_closes.keys())):
            price_row = prices_df[
                (prices_df["symbol"] == symbol) & (prices_df["date"] == td)
            ]
            if not price_row.empty:
                prev_closes[symbol] = float(price_row.iloc[0]["close"])

    # Realize remaining open positions for win-rate calculation
    for symbol, pnl in position_pnl.items():
        realized_position_pnls.append(pnl)

    metrics = _compute_metrics(
        equity_curve, daily_returns, all_trades, cfg.starting_capital, benchmark_returns,
        realized_position_pnls=realized_position_pnls,
    )

    # Compute forward_win_rate from realized_return populated by outcome_backfill.
    # win_rate above is always 0 for single-day sims (no hold period); this metric
    # is non-null once outcome_backfill runs (~5 trading days later).
    realized = predictions_df[predictions_df["realized_return"].notna()].copy()
    if not realized.empty:
        realized["correct"] = (
            ((realized["direction"] == "up") & (realized["realized_return"] > 0))
            | ((realized["direction"] == "down") & (realized["realized_return"] < 0))
        )
        metrics["forward_win_rate"] = round(float(realized["correct"].mean()), 4)
        metrics["forward_win_rate_n"] = int(len(realized))

    if daily_turnovers:
        metrics["avg_rebalance_turnover"] = round(float(np.mean(daily_turnovers)), 4)

    signal_health = _compute_signal_health(
        predictions_df, prices_df, trading_dates
    )
    metrics["ic_mean"] = signal_health["ic_mean"]
    metrics["ic_ir"] = signal_health["ic_ir"]
    metrics["decile_spread_mean_bps"] = signal_health["decile_spread_mean_bps"]
    metrics["ls_spread_mean_bps"] = signal_health["ls_spread_mean_bps"]
    metrics["signal_health"] = signal_health["signal_health_details"]

    _save_trades(db, all_trades)

    sim_run.status = "completed"
    sim_run.result_metrics = metrics
    db.commit()

    logger.info(
        "Simulation %s [%s] complete: return=%.2f%%, sharpe=%.3f, "
        "max_dd=%.2f%%, alpha=%.2f%%",
        run_id,
        strategy.name,
        metrics["total_return_pct"],
        metrics["sharpe_ratio"],
        metrics["max_drawdown_pct"],
        metrics.get("alpha_pct", 0),
    )

    return {
        "run_id": run_id,
        "strategy": strategy.name,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "n_trades": len(all_trades),
        "n_trading_days": len(trading_dates),
    }


def _compute_turnover(
    old_weights: dict[str, float], new_weights: dict[str, float]
) -> float:
    """Sum of absolute weight changes (0 = no change, 2 = full flip)."""
    all_symbols = set(list(old_weights.keys()) + list(new_weights.keys()))
    return sum(
        abs(new_weights.get(s, 0.0) - old_weights.get(s, 0.0)) for s in all_symbols
    )


# -----------------------------------------------------------------------
# Legacy entry point (backward-compatible)
# -----------------------------------------------------------------------


def run_simulation(
    db: Session,
    start_date: date,
    end_date: date,
    starting_capital: float = 100_000.0,
    min_confidence_trade: float = 0.65,
    max_position: float = 0.10,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    top_n: int = 10,
    model_version: str | None = None,
    enable_regime_guard: bool = True,
    vix_calm_below: float = 12.0,
    vix_elevated_below: float = 15.0,
    vix_stressed_above: float = 20.0,
    use_live_ic_guard: bool = False,
    live_ic_lookback_days: int = 60,
    live_ic_floor: float = 0.01,
    low_ic_exposure_scale: float = 0.5,
) -> dict:
    """Legacy simulation — kept for backward compatibility."""
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
        strategy_name="legacy_confidence_weighted",
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
    logger.info(
        f"Simulation {run_id} [legacy]: {len(trading_dates)} trading days, "
        f"capital=${starting_capital:,.0f}"
    )

    equity = starting_capital
    equity_curve = [{"date": str(start_date), "equity": equity}]
    all_trades: list[dict] = []
    daily_returns: list[float] = []
    benchmark_returns: list[float] = []
    daily_exposure_scales: list[float] = []
    live_ic_trace: list[float] = []

    cost_factor = (transaction_cost_bps + slippage_bps) / 10_000

    for td in trading_dates:
        day_preds = predictions_df[predictions_df["target_date"] == td].copy()

        day_preds = day_preds[day_preds["confidence"] >= min_confidence_trade]
        if day_preds.empty:
            equity_curve.append({"date": str(td), "equity": equity})
            daily_returns.append(0.0)
            benchmark_returns.append(_spy_return(prices_df, td))
            continue

        day_preds = day_preds.nlargest(top_n, "confidence")

        exposure_scale, live_ic = _compute_exposure_scale(
            db=db,
            day_preds=day_preds,
            as_of_date=td,
            model_version=model_version,
            enable_regime_guard=enable_regime_guard,
            vix_calm_below=vix_calm_below,
            vix_elevated_below=vix_elevated_below,
            vix_stressed_above=vix_stressed_above,
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
        day_trades: list[dict] = []

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
            direction_sign = 1.0 if pred["direction"] in ("up", "outperform") else -1.0
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
        benchmark_returns.append(_spy_return(prices_df, td))
        equity_curve.append({"date": str(td), "equity": round(equity, 2)})
        all_trades.extend(day_trades)

    metrics = _compute_metrics(
        equity_curve, daily_returns, all_trades, starting_capital, benchmark_returns
    )
    metrics["avg_exposure_scale"] = (
        round(float(np.mean(daily_exposure_scales)), 4)
        if daily_exposure_scales
        else 1.0
    )
    metrics["min_exposure_scale"] = (
        round(float(np.min(daily_exposure_scales)), 4)
        if daily_exposure_scales
        else 1.0
    )
    metrics["use_live_ic_guard"] = bool(use_live_ic_guard)
    metrics["live_ic_avg"] = (
        round(float(np.mean(live_ic_trace)), 5) if live_ic_trace else None
    )

    signal_health = _compute_signal_health(
        predictions_df, prices_df, trading_dates
    )
    metrics["ic_mean"] = signal_health["ic_mean"]
    metrics["ic_ir"] = signal_health["ic_ir"]
    metrics["decile_spread_mean_bps"] = signal_health["decile_spread_mean_bps"]
    metrics["ls_spread_mean_bps"] = signal_health["ls_spread_mean_bps"]
    metrics["signal_health"] = signal_health["signal_health_details"]

    _save_trades(db, all_trades)

    sim_run.status = "completed"
    sim_run.result_metrics = metrics
    db.commit()

    ic_str = f", IC={signal_health['ic_mean']}" if signal_health['ic_mean'] else ""
    ls_str = (
        f", LS_spread={signal_health['ls_spread_mean_bps']}bps"
        if signal_health['ls_spread_mean_bps'] else ""
    )
    logger.info(
        "Simulation %s [legacy] complete: return=%.2f%%, sharpe=%.3f, "
        "max_dd=%.2f%%, trades=%d%s%s",
        run_id,
        metrics["total_return_pct"],
        metrics["sharpe_ratio"],
        metrics["max_drawdown_pct"],
        metrics["total_trades"],
        ic_str,
        ls_str,
    )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "n_trades": len(all_trades),
        "n_trading_days": len(trading_dates),
    }


# -----------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------


def _load_predictions(
    db: Session, start_date: date, end_date: date, model_version: str | None
) -> pd.DataFrame:
    mv_filter = "AND p.model_version = :mv" if model_version else ""
    params: dict = {"start": start_date, "end": end_date}
    if model_version:
        params["mv"] = model_version

    result = db.execute(
        text(f"""
        SELECT p.symbol, p.target_date, p.direction, p.probability_up,
               p.confidence, p.model_version, fs.vix_level, p.realized_return
        FROM predictions p
        LEFT JOIN features_snapshot fs ON p.feature_snapshot_id = fs.snapshot_id
        WHERE target_date >= :start AND target_date <= :end
        {mv_filter}
        ORDER BY target_date, confidence DESC
    """),
        params,
    )

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "target_date",
            "direction",
            "probability_up",
            "confidence",
            "model_version",
            "vix_level",
            "realized_return",
        ],
    )
    # Normalize market-relative direction labels → strategy-canonical labels.
    # The production model uses "outperform"/"underperform"; strategies expect "up"/"down".
    df["direction"] = df["direction"].replace(
        {"outperform": "up", "underperform": "down"}
    )
    return df


def _load_features(
    db: Session, start_date: date, end_date: date
) -> pd.DataFrame:
    """Load features_snapshot rows for the simulation date range."""
    result = db.execute(
        text("""
        SELECT fs.symbol, fs.target_session_date,
               fs.rsi_14, fs.momentum_5d, fs.momentum_20d, fs.momentum_60d,
               fs.short_term_reversal,
               fs.sector_return_1d, fs.sector_return_5d,
               fs.sector_momentum_rank, fs.vix_level,
               fs.rolling_volatility_20d,
               s.sector
        FROM features_snapshot fs
        LEFT JOIN symbols s ON fs.symbol = s.symbol
        WHERE fs.target_session_date >= :start
          AND fs.target_session_date <= :end
        ORDER BY fs.target_session_date
    """),
        {"start": start_date, "end": end_date},
    )
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "target_session_date",
            "rsi_14",
            "momentum_5d",
            "momentum_20d",
            "momentum_60d",
            "short_term_reversal",
            "sector_return_1d",
            "sector_return_5d",
            "sector_momentum_rank",
            "vix_level",
            "rolling_volatility_20d",
            "sector",
        ],
    )


def _load_prices(db: Session, start_date: date, end_date: date) -> pd.DataFrame:
    result = db.execute(
        text("""
        SELECT symbol, date, open, close, volume
        FROM market_bars_daily
        WHERE date >= :start AND date <= :end
        ORDER BY date
    """),
        {"start": start_date, "end": end_date},
    )

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=["symbol", "date", "open", "close", "volume"])


def _load_market_window(db: Session, end_date: date, lookback: int = 60) -> pd.DataFrame:
    """Load recent market bars for ADV computation."""
    result = db.execute(
        text("""
        SELECT symbol, date, open, high, low, close, volume
        FROM market_bars_daily
        WHERE date >= :start AND date <= :end
        ORDER BY date
    """),
        {"start": end_date - timedelta(days=lookback * 2), "end": end_date},
    )
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"]
    )


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _get_vix(features_df: pd.DataFrame) -> float | None:
    if features_df.empty or "vix_level" not in features_df.columns:
        return None
    vals = features_df["vix_level"].dropna()
    return float(vals.median()) if not vals.empty else None


def _spy_return(prices_df: pd.DataFrame, td: date) -> float:
    spy = prices_df[(prices_df["symbol"] == "SPY") & (prices_df["date"] == td)]
    if spy.empty:
        return 0.0
    o = float(spy.iloc[0]["open"])
    c = float(spy.iloc[0]["close"])
    return (c - o) / o if o > 0 else 0.0


def _compute_weights(
    preds_df: pd.DataFrame, max_position: float, exposure_scale: float = 1.0
) -> dict[str, float]:
    """Confidence-weighted allocation with position cap."""
    total_confidence = preds_df["confidence"].sum()
    if total_confidence == 0:
        return {}

    weights: dict[str, float] = {}
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
    vix_calm_below: float,
    vix_elevated_below: float,
    vix_stressed_above: float,
    use_live_ic_guard: bool,
    live_ic_lookback_days: int,
    live_ic_floor: float,
    low_ic_exposure_scale: float,
) -> tuple[float, float | None]:
    """Combine VIX-based and live-IC-based exposure controls (legacy).

    VIX regime mapping (standard risk-off behaviour):
      VIX < calm_below  (12):  calm market    → full exposure  (1.00)
      VIX < elevated    (15):  moderate       → 75% exposure
      VIX < stressed    (20):  elevated       → 50% exposure
      VIX >= stressed   (20):  stressed/crisis → 25% exposure
    """
    if not enable_regime_guard:
        return 1.0, None

    vix_scale = 1.0
    if "vix_level" in day_preds.columns and day_preds["vix_level"].notna().any():
        vix = float(day_preds["vix_level"].median())
        if vix < vix_calm_below:
            vix_scale = 1.0
        elif vix < vix_elevated_below:
            vix_scale = 0.75
        elif vix >= vix_stressed_above:
            vix_scale = 0.25
        else:
            vix_scale = 0.50

    live_ic = _compute_live_rolling_ic(
        db, as_of_date, live_ic_lookback_days, model_version=model_version
    )

    ic_scale = 1.0
    if use_live_ic_guard and live_ic is not None and live_ic < live_ic_floor:
        ic_scale = float(min(max(low_ic_exposure_scale, 0.0), 1.0))

    return min(vix_scale, ic_scale), live_ic


def _compute_live_rolling_ic(
    db: Session,
    as_of_date: date,
    lookback_days: int,
    model_version: str | None = None,
) -> float | None:
    """Compute trailing mean daily IC from realized predictions."""
    mv_filter = "AND model_version = :mv" if model_version else ""
    params: dict = {
        "start": as_of_date - timedelta(days=lookback_days),
        "end": as_of_date,
    }
    if model_version:
        params["mv"] = model_version

    result = db.execute(
        text(f"""
        SELECT target_date, probability_up, realized_return
        FROM predictions
        WHERE target_date >= :start
          AND target_date < :end
          AND realized_return IS NOT NULL
          {mv_filter}
        ORDER BY target_date
    """),
        params,
    )
    rows = result.fetchall()
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["target_date", "probability_up", "realized_return"])
    if df.empty:
        return None
    daily_ics: list[float] = []
    for _, day_df in df.groupby("target_date"):
        if len(day_df) < 10:
            continue
        ic = day_df["probability_up"].corr(day_df["realized_return"], method="spearman")
        if pd.notna(ic):
            daily_ics.append(float(ic))
    if not daily_ics:
        return None
    return float(np.mean(daily_ics))


def _compute_signal_health(
    all_preds: pd.DataFrame,
    prices_df: pd.DataFrame,
    trading_dates: list,
    n_deciles: int = 10,
) -> dict:
    """Compute daily IC, decile spread, and long-short spread.

    Uses close-to-close returns (matching the strategy framework's P&L
    definition) and the FULL cross-section of predictions, not just traded
    symbols, because signal health measures ranking ability over the universe.
    """
    daily_ics: list[float] = []
    daily_ls_spreads_bps: list[float] = []
    daily_details: list[dict] = []

    prices_sorted = prices_df.sort_values(["symbol", "date"])
    prices_sorted["prev_close"] = prices_sorted.groupby("symbol")["close"].shift(1)
    prices_sorted["cc_return"] = (
        (prices_sorted["close"] - prices_sorted["prev_close"])
        / prices_sorted["prev_close"]
    )

    for td in trading_dates:
        day_preds = all_preds[all_preds["target_date"] == td].copy()
        if len(day_preds) < 20:
            continue

        day_returns = prices_sorted[prices_sorted["date"] == td][
            ["symbol", "cc_return"]
        ]
        day_preds = day_preds.merge(day_returns, on="symbol", how="left")
        day_preds = day_preds.rename(columns={"cc_return": "realized_ret"})
        day_preds = day_preds.dropna(subset=["realized_ret", "probability_up"])

        if len(day_preds) < 20:
            continue

        ic = float(day_preds["probability_up"].corr(
            day_preds["realized_ret"], method="spearman"
        ))
        if pd.notna(ic):
            daily_ics.append(ic)

        day_preds["decile"] = pd.qcut(
            day_preds["probability_up"], n_deciles, labels=False, duplicates="drop"
        )
        decile_returns = day_preds.groupby("decile")["realized_ret"].mean()
        if len(decile_returns) >= 2:
            top_ret = float(decile_returns.iloc[-1])
            bot_ret = float(decile_returns.iloc[0])
            ls_bps = (top_ret - bot_ret) * 10_000
            daily_ls_spreads_bps.append(ls_bps)

            daily_details.append({
                "date": str(td),
                "ic": round(ic, 5) if pd.notna(ic) else None,
                "top_decile_ret_bps": round(top_ret * 10_000, 2),
                "bot_decile_ret_bps": round(bot_ret * 10_000, 2),
                "ls_spread_bps": round(ls_bps, 2),
                "n_symbols": len(day_preds),
            })

    result: dict = {
        "daily_ic_trace": [round(x, 5) for x in daily_ics],
        "ic_mean": round(float(np.mean(daily_ics)), 5) if daily_ics else None,
        "ic_std": round(float(np.std(daily_ics)), 5) if daily_ics else None,
        "ic_ir": (
            round(float(np.mean(daily_ics) / np.std(daily_ics)), 4)
            if daily_ics and np.std(daily_ics) > 0 else None
        ),
        "decile_spread_mean_bps": (
            round(float(np.mean(daily_ls_spreads_bps)), 2)
            if daily_ls_spreads_bps else None
        ),
        "ls_spread_mean_bps": (
            round(float(np.mean(daily_ls_spreads_bps)), 2)
            if daily_ls_spreads_bps else None
        ),
        "ls_spread_daily_bps": (
            [round(x, 2) for x in daily_ls_spreads_bps]
            if daily_ls_spreads_bps else []
        ),
        "signal_health_details": daily_details,
    }
    return result


def _newey_west_sharpe(returns: np.ndarray, lag: int = 4) -> float:
    """Newey-West adjusted Sharpe for serially correlated daily returns."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return 0.0
    base_sr = float((np.mean(arr) / std) * np.sqrt(252))
    if lag <= 0:
        return base_sr
    x = arr - float(np.mean(arr))
    var0 = float(np.dot(x, x) / (n - 1))
    if var0 <= 0:
        return base_sr
    rho_sum = 0.0
    for k in range(1, min(lag, n - 1) + 1):
        gamma_k = float(np.dot(x[k:], x[:-k]) / (n - 1))
        weight = 1.0 - (k / (lag + 1.0))
        rho_sum += weight * (gamma_k / var0)
    adj = np.sqrt(max(1e-12, 1.0 + 2.0 * rho_sum))
    return float(base_sr / adj)


def _compute_metrics(
    equity_curve: list[dict],
    daily_returns: list[float],
    trades: list[dict],
    starting_capital: float,
    benchmark_returns: list[float] | None = None,
    nw_lag: int = 4,
    realized_position_pnls: list[float] | None = None,
) -> dict:
    """Compute simulation performance metrics with Newey-West adjusted Sharpe.

    For strategy-framework simulations, trades are recorded at rebalance
    with ``daily_pnl=0``.  In that case *realized_position_pnls* (P&L per
    round-trip position) is used to compute a meaningful win rate.
    """
    final_equity = equity_curve[-1]["equity"] if equity_curve else starting_capital
    total_return = (final_equity - starting_capital) / starting_capital

    returns = np.array(daily_returns) if daily_returns else np.array([0.0])
    trading_days = len(returns)

    annualized_return = (
        ((1 + total_return) ** (252 / max(trading_days, 1)) - 1) * 100
    )

    sharpe = 0.0
    if returns.std() > 0 and trading_days > 1:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    sharpe_nw = _newey_west_sharpe(returns, lag=nw_lag)

    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if realized_position_pnls:
        wins = sum(1 for p in realized_position_pnls if p > 0)
        total_trades = len(realized_position_pnls)
    else:
        wins = sum(1 for t in trades if t["daily_pnl"] > 0)
        total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0

    total_costs = sum(t["transaction_cost"] + t.get("slippage_cost", 0) for t in trades)

    unique_days = len(set(t["date"] for t in trades)) if trades else 0
    avg_daily_turnover = total_trades / unique_days if unique_days > 0 else 0

    result = {
        "total_return_pct": round(total_return * 100, 4),
        "annualized_return_pct": round(annualized_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sharpe_ratio_nw": round(sharpe_nw, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "win_rate": round(win_rate, 4),
        "total_trades": total_trades,
        "total_trading_days": trading_days,
        "total_transaction_costs": round(total_costs, 2),
        "avg_daily_turnover": round(avg_daily_turnover, 2),
        "final_equity": round(final_equity, 2),
        "net_profit": round(final_equity - starting_capital, 2),
    }

    # Benchmark comparison
    if benchmark_returns:
        bench = np.array(benchmark_returns)
        cum_bench = float(np.prod(1 + bench) - 1)
        result["benchmark_return_pct"] = round(cum_bench * 100, 4)
        result["alpha_pct"] = round((total_return - cum_bench) * 100, 4)

        if len(bench) > 1 and bench.std() > 0:
            cov = np.cov(returns[: len(bench)], bench)[0, 1]
            beta = cov / bench.var() if bench.var() > 0 else 0
            result["beta"] = round(float(beta), 4)

            tracking_error = float(np.std(returns[: len(bench)] - bench, ddof=1))
            if tracking_error > 0:
                result["information_ratio"] = round(
                    float(np.mean(returns[: len(bench)] - bench))
                    / tracking_error
                    * np.sqrt(252),
                    4,
                )

    return result


# -----------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------


def _sanitize_trade(trade: dict) -> dict:
    """Convert numpy scalars to native Python types for DB insertion."""
    clean = {}
    for k, v in trade.items():
        if isinstance(v, (np.floating, np.integer)):
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean


def _save_trades(db: Session, trades: list[dict]) -> int:
    if not trades:
        return 0

    sanitized = [_sanitize_trade(t) for t in trades]
    db.bulk_insert_mappings(PaperTrade, sanitized)
    db.commit()
    return len(trades)
