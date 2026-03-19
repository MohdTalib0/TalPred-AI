"""Shared risk management layer.

Applied after signal generation and before portfolio construction.
Enforces position limits, sector limits, side limits, volatility
targeting (EWMA or rolling), factor exposure constraints, ADV
participation caps, and VIX regime scaling.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.strategies.base import Signal
from src.strategies.config import RiskConfig

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        config: RiskConfig | None = None,
        factor_model=None,
    ) -> None:
        self.cfg = config or RiskConfig()
        self._factor_model = factor_model

    def set_factor_model(self, factor_model) -> None:
        """Attach a fitted StatisticalFactorModel for factor constraints.

        Automatically enables factor_constraint_enabled when a model is
        attached, so callers don't need to remember to flip the flag.
        """
        self._factor_model = factor_model
        if factor_model is not None:
            self.cfg.factor_constraint_enabled = True

    def apply(
        self,
        signals: list[Signal],
        features_df: pd.DataFrame,
        market_df: pd.DataFrame,
        equity: float,
        daily_returns: list[float],
        vix: float | None = None,
    ) -> list[Signal]:
        """Filter and constrain signals through the full risk stack.

        Returns a (potentially smaller) list of signals that survive all
        risk checks.
        """
        if not signals:
            return []

        signals = self._drawdown_breaker(signals, daily_returns)
        if not signals:
            return []

        signals = self._adv_filter(signals, market_df, equity)
        signals = self._min_weight_filter(signals)
        signals = self._position_cap(signals)
        signals = self._sector_cap(signals, features_df)
        signals = self._side_limits(signals)
        signals = self._vix_scale(signals, vix)
        signals = self._vol_target(signals, daily_returns)

        if self.cfg.factor_constraint_enabled and self._factor_model is not None:
            signals = self._factor_exposure_constraint(signals)

        return signals

    # ------------------------------------------------------------------
    # Individual risk checks
    # ------------------------------------------------------------------

    def _drawdown_breaker(
        self, signals: list[Signal], daily_returns: list[float]
    ) -> list[Signal]:
        """Kill or reduce exposure when trailing returns breach drawdown thresholds."""
        n = self.cfg.drawdown_lookback_days
        if len(daily_returns) < n:
            return signals

        trailing = np.array(daily_returns[-n:])
        cum_ret = float(np.prod(1 + trailing) - 1)

        if cum_ret <= self.cfg.drawdown_halt_threshold:
            logger.warning(
                "DRAWDOWN HALT: %d-day return=%.2f%% <= %.2f%% → going flat",
                n, cum_ret * 100, self.cfg.drawdown_halt_threshold * 100,
            )
            return []

        if cum_ret <= self.cfg.drawdown_reduce_threshold:
            scale = self.cfg.drawdown_reduce_scale
            logger.warning(
                "DRAWDOWN REDUCE: %d-day return=%.2f%% <= %.2f%% → scaling to %.0f%%",
                n, cum_ret * 100, self.cfg.drawdown_reduce_threshold * 100, scale * 100,
            )
            for sig in signals:
                sig.raw_score *= scale
            return signals

        return signals

    def _adv_filter(
        self, signals: list[Signal], market_df: pd.DataFrame, equity: float
    ) -> list[Signal]:
        """Drop signals where notional would exceed ADV participation limit."""
        if market_df.empty:
            return signals

        adv_lookup = self._compute_adv(market_df)
        passed: list[Signal] = []
        for sig in signals:
            adv = adv_lookup.get(sig.symbol, 0)
            if adv <= 0:
                logger.debug("ADV filter dropped %s: ADV is zero/missing", sig.symbol)
                continue
            notional = equity * self.cfg.max_position_weight
            if notional / adv > self.cfg.adv_participation_limit:
                logger.debug(
                    "ADV filter dropped %s: notional=$%.0f vs ADV=$%.0f (%.2f%%)",
                    sig.symbol,
                    notional,
                    adv,
                    notional / adv * 100,
                )
                continue
            passed.append(sig)
        return passed

    def _min_weight_filter(self, signals: list[Signal]) -> list[Signal]:
        """Drop signals with absolute score below min weight threshold."""
        if not signals:
            return signals
        max_score = max(abs(s.raw_score) for s in signals)
        if max_score == 0:
            return []
        return [
            s
            for s in signals
            if abs(s.raw_score) / max_score >= self.cfg.min_weight_threshold / self.cfg.max_position_weight
        ]

    def _position_cap(self, signals: list[Signal]) -> list[Signal]:
        """Normalize so no single signal exceeds max_position_weight."""
        if not signals:
            return signals
        total_abs = sum(abs(s.raw_score) for s in signals)
        if total_abs == 0:
            return []
        for sig in signals:
            proposed = abs(sig.raw_score) / total_abs
            if proposed > self.cfg.max_position_weight:
                scale = self.cfg.max_position_weight / proposed
                sig.raw_score *= scale
        return signals

    def _sector_cap(
        self, signals: list[Signal], features_df: pd.DataFrame
    ) -> list[Signal]:
        """Trim signals so no sector exceeds max_sector_weight of gross exposure."""
        sector_map: dict[str, str] = {}
        if not features_df.empty and "symbol" in features_df.columns and "sector" in features_df.columns:
            raw = features_df.set_index("symbol")["sector"].to_dict()
            sector_map = {k: v for k, v in raw.items() if v is not None}
            n_missing = len(raw) - len(sector_map)
            if n_missing > 0:
                logger.debug(
                    "Sector cap: %d/%d symbols have NULL sector — excluded from cap",
                    n_missing, len(raw),
                )

        if not sector_map:
            return signals

        total_abs = sum(abs(s.raw_score) for s in signals)
        if total_abs == 0:
            return signals

        sector_weight: dict[str, float] = {}
        for sig in signals:
            sector = sector_map.get(sig.symbol, "Unknown")
            w = abs(sig.raw_score) / total_abs
            sector_weight.setdefault(sector, 0.0)
            sector_weight[sector] += w

        exceeded = {s for s, w in sector_weight.items() if w > self.cfg.max_sector_weight}
        if not exceeded:
            return signals

        result: list[Signal] = []
        for sig in signals:
            sector = sector_map.get(sig.symbol, "Unknown")
            if sector in exceeded:
                excess_ratio = self.cfg.max_sector_weight / sector_weight[sector]
                sig.raw_score *= excess_ratio
            result.append(sig)
        return result

    def _side_limits(self, signals: list[Signal]) -> list[Signal]:
        """Scale long and short sides to respect gross limits."""
        if not signals:
            return signals

        total_abs = sum(abs(s.raw_score) for s in signals)
        if total_abs == 0:
            return []

        long_total = sum(abs(s.raw_score) for s in signals if s.direction == "long")
        short_total = sum(abs(s.raw_score) for s in signals if s.direction == "short")

        long_frac = long_total / total_abs
        short_frac = short_total / total_abs

        long_scale = min(1.0, self.cfg.max_gross_long / long_frac) if long_frac > 0 else 1.0
        short_scale = min(1.0, self.cfg.max_gross_short / short_frac) if short_frac > 0 else 1.0

        for sig in signals:
            if sig.direction == "long":
                sig.raw_score *= long_scale
            else:
                sig.raw_score *= short_scale
        return signals

    def _vix_scale(self, signals: list[Signal], vix: float | None) -> list[Signal]:
        """Reduce exposure in high-VIX regimes (corrected: high VIX = less exposure)."""
        if vix is None:
            return signals

        if vix >= self.cfg.vix_crisis:
            scale = 0.25
        elif vix >= self.cfg.vix_elevated:
            scale = 0.50
        elif vix >= self.cfg.vix_above_avg:
            scale = 0.75
        else:
            scale = 1.0

        if scale < 1.0:
            logger.info("VIX regime scale: VIX=%.1f -> exposure=%.0f%%", vix, scale * 100)
            for sig in signals:
                sig.raw_score *= scale
        return signals

    def _vol_target(
        self, signals: list[Signal], daily_returns: list[float]
    ) -> list[Signal]:
        """Scale exposure so portfolio annualized vol targets vol_target_annual.

        Uses EWMA volatility (half-life=10d) when use_ewma_vol is True,
        otherwise falls back to simple rolling std. EWMA reacts faster
        to regime changes than a flat 20-day window.
        """
        min_obs = max(self.cfg.vol_lookback_days, self.cfg.ewma_halflife_days * 2)
        if len(daily_returns) < min_obs:
            return signals

        if self.cfg.use_ewma_vol:
            realized_vol = self._ewma_vol(daily_returns)
        else:
            recent = np.array(daily_returns[-self.cfg.vol_lookback_days:])
            realized_vol = float(np.std(recent, ddof=1)) * np.sqrt(252)

        if realized_vol <= 0:
            return signals

        scale = self.cfg.vol_target_annual / realized_vol
        scale = min(scale, self.cfg.vol_scale_cap)
        scale = max(scale, 0.1)

        vol_method = "EWMA" if self.cfg.use_ewma_vol else "rolling"
        if abs(scale - 1.0) > 0.05:
            logger.info(
                "Vol-target (%s): realized=%.1f%% target=%.1f%% -> scale=%.2f",
                vol_method,
                realized_vol * 100,
                self.cfg.vol_target_annual * 100,
                scale,
            )
            for sig in signals:
                sig.raw_score *= scale
        return signals

    def _ewma_vol(self, daily_returns: list[float]) -> float:
        """Compute annualized EWMA volatility with configurable half-life.

        EWMA variance: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t
        where λ = exp(-ln(2) / half_life)
        """
        halflife = self.cfg.ewma_halflife_days
        decay = np.exp(-np.log(2) / halflife)

        returns = np.array(daily_returns)
        warmup = min(5, len(returns))
        ewma_var = float(np.mean(returns[:warmup] ** 2))
        for r in returns[warmup:]:
            ewma_var = decay * ewma_var + (1 - decay) * r ** 2

        return float(np.sqrt(ewma_var) * np.sqrt(252))

    def _factor_exposure_constraint(self, signals: list[Signal]) -> list[Signal]:
        """Scale down signals when portfolio factor exposure exceeds limit.

        Uses the attached StatisticalFactorModel to compute the portfolio's
        exposure to each PCA factor. If any factor exposure exceeds
        max_factor_exposure, scales all signals proportionally.
        """
        if not signals or self._factor_model is None:
            return signals

        total_abs = sum(abs(s.raw_score) for s in signals)
        if total_abs == 0:
            return signals

        weights = {}
        for sig in signals:
            w = sig.raw_score / total_abs
            weights[sig.symbol] = weights.get(sig.symbol, 0.0) + w

        try:
            exposure = self._factor_model.portfolio_exposure(weights)
        except RuntimeError:
            return signals

        max_exp = float(np.max(np.abs(exposure)))
        limit = self.cfg.max_factor_exposure

        if max_exp > limit:
            scale = limit / max_exp
            worst_factor = int(np.argmax(np.abs(exposure)))
            logger.warning(
                "Factor constraint: F%d exposure=%.2f > limit=%.2f → "
                "scaling signals by %.2f",
                worst_factor + 1, max_exp, limit, scale,
            )
            for sig in signals:
                sig.raw_score *= scale

        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_adv(market_df: pd.DataFrame, lookback: int = 30) -> dict[str, float]:
        """30-day average dollar volume per symbol."""
        if market_df.empty:
            return {}
        df = market_df.copy()
        if "dollar_volume" not in df.columns:
            if "close" in df.columns and "volume" in df.columns:
                df["dollar_volume"] = df["close"].abs() * df["volume"].abs()
            else:
                return {}

        recent = df.sort_values("date", ascending=False).groupby("symbol").head(lookback)
        return recent.groupby("symbol")["dollar_volume"].mean().to_dict()
