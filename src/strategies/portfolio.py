"""Portfolio constructor.

Converts risk-adjusted signals into trade records with realistic cost
modeling, benchmark (SPY) tracking, beta-neutrality, and rebalance-stride
support.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.strategies.base import Signal
from src.strategies.config import CostConfig

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    def __init__(
        self,
        cost_config: CostConfig | None = None,
        beta_neutral: bool = False,
        turnover_penalty: float = 0.0,
    ) -> None:
        self.cost = cost_config or CostConfig()
        self.beta_neutral = beta_neutral
        self.turnover_penalty = turnover_penalty

    def compute_target_weights(
        self,
        signals: list[Signal],
        market_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        target_date: date,
        current_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Convert signals to target portfolio weights (with optional beta hedge).

        When ``turnover_penalty > 0`` and ``current_weights`` is provided,
        the new weights are blended toward current holdings to reduce
        unnecessary turnover.  The penalty acts as a shrinkage parameter:
        ``w_final = (1 - λ) * w_ideal + λ * w_current`` (normalized).

        Returns signed weight dict (positive=long, negative=short),
        normalized to sum(abs) = 1.
        """
        if not signals:
            return {}
        weights = self._signals_to_weights(signals)
        if self.beta_neutral:
            weights = self._beta_neutralize(
                weights, market_df, prices_df, target_date
            )

        if self.turnover_penalty > 0 and current_weights:
            weights = self._apply_turnover_penalty(weights, current_weights)

        return weights

    def _apply_turnover_penalty(
        self,
        ideal_weights: dict[str, float],
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """Blend ideal weights toward current holdings to reduce turnover.

        Uses shrinkage: w = (1 - λ) * w_ideal + λ * w_current, then
        re-normalizes so sum(|w|) = 1.
        """
        lam = min(max(self.turnover_penalty, 0.0), 0.9)
        all_symbols = set(list(ideal_weights.keys()) + list(current_weights.keys()))
        blended: dict[str, float] = {}
        for sym in all_symbols:
            w_ideal = ideal_weights.get(sym, 0.0)
            w_curr = current_weights.get(sym, 0.0)
            blended[sym] = (1 - lam) * w_ideal + lam * w_curr

        total_abs = sum(abs(v) for v in blended.values())
        if total_abs > 0:
            blended = {k: v / total_abs for k, v in blended.items()}
        return {k: v for k, v in blended.items() if abs(v) > 1e-8}

    def compute_rebalance_cost(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        equity: float,
        adv_lookup: dict[str, float],
    ) -> float:
        """Compute total transaction cost for the weight delta (turnover).

        Costs are only charged on the changed portion, not on the full
        portfolio — this is the key difference from daily round-trip.
        """
        all_symbols = set(list(old_weights.keys()) + list(new_weights.keys()))
        total_cost = 0.0
        for symbol in all_symbols:
            delta = abs(new_weights.get(symbol, 0.0) - old_weights.get(symbol, 0.0))
            if delta < 1e-8:
                continue
            trade_value = equity * delta
            adv = adv_lookup.get(symbol, 0)
            slip = self.cost.slippage_bps(trade_value, adv)
            total_cost += trade_value * (self.cost.base_cost_bps + slip) / 10_000
        return total_cost

    def apply_partial_fills(
        self,
        weights: dict[str, float],
        equity: float,
        adv_lookup: dict[str, float],
    ) -> dict[str, float]:
        """Clip weights so no position exceeds max_participation * ADV.

        If a $125K position requires trading 10% of a stock's daily volume,
        but max_participation is 5%, the weight is halved. This prevents
        unrealistic fills in illiquid small-cap names.
        """
        max_part = self.cost.max_participation_rate
        if max_part <= 0 or max_part >= 1.0:
            return weights

        clipped: dict[str, float] = {}
        for sym, w in weights.items():
            adv = adv_lookup.get(sym, 0)
            if adv <= 0:
                continue
            trade_value = equity * abs(w)
            participation = trade_value / adv
            if participation > max_part:
                scale = max_part / participation
                clipped[sym] = w * scale
                logger.debug(
                    "Partial fill %s: participation %.1f%% > %.1f%%, "
                    "weight %.4f → %.4f",
                    sym, participation * 100, max_part * 100, w, clipped[sym],
                )
            else:
                clipped[sym] = w
        return {k: v for k, v in clipped.items() if abs(v) > 1e-8}

    def build_trades(
        self,
        signals: list[Signal],
        prices_df: pd.DataFrame,
        market_df: pd.DataFrame,
        equity: float,
        target_date: date,
        run_id: str,
    ) -> tuple[list[dict], float, float]:
        """Convert signals to executable trade records (single-day O->C).

        Designed for legacy single-day simulations where each position is
        entered at open and exited at close on the same day.  Transaction
        costs are charged as round-trip (entry + exit, ``* 2``).

        For multi-day hold strategies, use ``compute_target_weights`` +
        ``compute_rebalance_cost`` instead — those charge costs only on
        the turnover delta between rebalance days.

        Returns:
            (trades, day_pnl, benchmark_return)
        """
        if not signals:
            benchmark_ret = self._benchmark_return(prices_df, target_date)
            return [], 0.0, benchmark_ret

        weights = self._signals_to_weights(signals)

        if self.beta_neutral:
            weights = self._beta_neutralize(weights, market_df, prices_df, target_date)

        adv_lookup = self._adv_lookup(market_df)

        trades: list[dict] = []
        day_pnl = 0.0

        for symbol, weight in weights.items():
            price_row = prices_df[
                (prices_df["symbol"] == symbol) & (prices_df["date"] == target_date)
            ]
            if price_row.empty:
                continue

            open_price = float(price_row.iloc[0]["open"])
            close_price = float(price_row.iloc[0]["close"])
            if open_price <= 0:
                continue

            direction = "long" if weight > 0 else "short"
            abs_weight = abs(weight)
            position_value = equity * abs_weight

            adv = adv_lookup.get(symbol, 0)
            slip_bps = self.cost.slippage_bps(position_value, adv)

            if direction == "long":
                entry = open_price * (1 + slip_bps / 10_000)
                exit_ = close_price * (1 - slip_bps / 10_000)
                direction_sign = 1.0
            else:
                entry = open_price * (1 - slip_bps / 10_000)
                exit_ = close_price * (1 + slip_bps / 10_000)
                direction_sign = -1.0

            shares = position_value / open_price
            gross_pnl = direction_sign * shares * (close_price - open_price)

            tx_cost = position_value * (self.cost.base_cost_bps / 10_000) * 2
            slip_cost = position_value * (slip_bps / 10_000) * 2

            borrow_cost = 0.0
            if direction == "short":
                borrow_cost = self.cost.daily_borrow_cost(position_value)

            net_pnl = gross_pnl - tx_cost - slip_cost - borrow_cost
            day_pnl += net_pnl

            trades.append(
                {
                    "run_id": run_id,
                    "date": target_date,
                    "symbol": symbol,
                    "weight": weight,
                    "position_qty": float(shares * direction_sign),
                    "entry_price": float(entry),
                    "exit_price": float(exit_),
                    "transaction_cost": float(tx_cost),
                    "slippage_cost": float(slip_cost + borrow_cost),
                    "daily_pnl": float(net_pnl),
                }
            )

        benchmark_ret = self._benchmark_return(prices_df, target_date)
        return trades, float(day_pnl), benchmark_ret

    @staticmethod
    def _signals_to_weights(signals: list[Signal]) -> dict[str, float]:
        """Convert signals to signed weight dict normalized to sum(abs) = 1."""
        total = sum(abs(s.raw_score) for s in signals)
        if total == 0:
            return {}
        weights: dict[str, float] = {}
        for sig in signals:
            w = sig.raw_score / total
            if sig.symbol in weights:
                weights[sig.symbol] += w
            else:
                weights[sig.symbol] = w
        return {k: v for k, v in weights.items() if abs(v) > 1e-8}

    @staticmethod
    def _benchmark_return(prices_df: pd.DataFrame, target_date: date) -> float:
        """SPY open-to-close return for the day."""
        spy = prices_df[
            (prices_df["symbol"] == "SPY") & (prices_df["date"] == target_date)
        ]
        if spy.empty:
            return 0.0
        o = float(spy.iloc[0]["open"])
        c = float(spy.iloc[0]["close"])
        return (c - o) / o if o > 0 else 0.0

    @staticmethod
    def _adv_lookup(market_df: pd.DataFrame, lookback: int = 30) -> dict[str, float]:
        """30-day average dollar volume."""
        if market_df.empty:
            return {}
        df = market_df.copy()
        if "dollar_volume" not in df.columns:
            if "close" in df.columns and "volume" in df.columns:
                df["dollar_volume"] = df["close"].abs() * df["volume"].abs()
            else:
                return {}
        recent = df.sort_values("date", ascending=False).groupby("symbol").head(lookback)
        adv = recent.groupby("symbol")["dollar_volume"].mean()
        return adv[adv > 0].to_dict()

    @staticmethod
    def _estimate_betas(
        market_df: pd.DataFrame,
        symbols: list[str],
        lookback: int = 60,
    ) -> dict[str, float]:
        """Estimate per-stock betas with Vasicek (Bayesian) shrinkage.

        Raw OLS betas from a short window are noisy. Vasicek shrinkage
        blends each raw beta toward the cross-sectional mean, weighted
        by the precision (inverse variance) of each estimate:

            β_shrunk = (σ²_cs * β_raw + σ²_raw * β_cs_mean)
                       / (σ²_cs + σ²_raw)

        where σ²_raw is the standard error of the OLS beta estimate and
        σ²_cs is the cross-sectional variance of raw betas.
        """
        spy_rets = (
            market_df[market_df["symbol"] == "SPY"]
            .sort_values("date")
            .tail(lookback)
            .assign(ret=lambda d: d["close"].pct_change())
        )
        if len(spy_rets.dropna()) < 20:
            return {}

        spy_ret_series = spy_rets.set_index("date")["ret"].dropna()

        raw_betas: dict[str, float] = {}
        beta_se: dict[str, float] = {}

        for symbol in symbols:
            if symbol == "SPY":
                raw_betas[symbol] = 1.0
                beta_se[symbol] = 0.0
                continue

            sym_rets = (
                market_df[market_df["symbol"] == symbol]
                .sort_values("date")
                .tail(lookback)
                .assign(ret=lambda d: d["close"].pct_change())
                .set_index("date")["ret"]
                .dropna()
            )
            common = spy_ret_series.index.intersection(sym_rets.index)
            if len(common) < 20:
                raw_betas[symbol] = 1.0
                beta_se[symbol] = 1.0
                continue

            spy_aligned = spy_ret_series.loc[common].values
            sym_aligned = sym_rets.loc[common].values
            spy_var = np.var(spy_aligned, ddof=1)

            if spy_var < 1e-12:
                raw_betas[symbol] = 1.0
                beta_se[symbol] = 1.0
                continue

            cov_mat = np.cov(sym_aligned, spy_aligned, ddof=1)
            beta_raw = cov_mat[0, 1] / spy_var
            raw_betas[symbol] = float(beta_raw)

            residual = sym_aligned - beta_raw * spy_aligned
            n = len(common)
            residual_var = np.var(residual, ddof=2) if n > 2 else 1.0
            se_squared = residual_var / (spy_var * n) if spy_var * n > 0 else 1.0
            beta_se[symbol] = float(se_squared)

        if not raw_betas:
            return {}

        # Cross-sectional prior: mean and variance of raw betas
        beta_vals = [v for k, v in raw_betas.items() if k != "SPY"]
        if not beta_vals:
            return raw_betas

        cs_mean = float(np.mean(beta_vals))
        cs_var = float(np.var(beta_vals, ddof=1)) if len(beta_vals) > 1 else 0.25

        shrunk_betas: dict[str, float] = {}
        for symbol in raw_betas:
            if symbol == "SPY":
                shrunk_betas[symbol] = 1.0
                continue

            se_sq = beta_se.get(symbol, 1.0)
            if cs_var + se_sq < 1e-12:
                shrunk_betas[symbol] = cs_mean
            else:
                shrunk_betas[symbol] = (
                    cs_var * raw_betas[symbol] + se_sq * cs_mean
                ) / (cs_var + se_sq)

        return shrunk_betas

    @staticmethod
    def _beta_neutralize(
        weights: dict[str, float],
        market_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        target_date: date,
        lookback: int = 60,
    ) -> dict[str, float]:
        """Adjust weights to make portfolio beta approximately zero.

        Uses Vasicek-shrunk betas for robust estimation and adds a SPY
        hedge so that sum(w_i * beta_i) ~ 0.
        """
        if market_df.empty:
            return weights

        stock_symbols = [s for s in weights if s != "SPY"]
        if not stock_symbols:
            return weights

        betas = PortfolioConstructor._estimate_betas(
            market_df, stock_symbols, lookback=lookback
        )
        if not betas:
            logger.warning("Insufficient data for beta estimation, skipping neutralization")
            return weights

        portfolio_beta = 0.0
        for symbol, w in weights.items():
            if symbol == "SPY":
                continue
            portfolio_beta += w * betas.get(symbol, 1.0)

        if abs(portfolio_beta) < 0.01:
            return weights

        hedge_weight = -portfolio_beta
        result = dict(weights)
        result["SPY"] = result.get("SPY", 0.0) + hedge_weight

        total_abs = sum(abs(v) for v in result.values())
        if total_abs > 0:
            result = {k: v / total_abs for k, v in result.items()}

        logger.info(
            f"Beta hedge (Vasicek-shrunk): portfolio_beta={portfolio_beta:.3f}, "
            f"SPY_hedge_weight={hedge_weight:.3f}"
        )
        return {k: v for k, v in result.items() if abs(v) > 1e-8}
