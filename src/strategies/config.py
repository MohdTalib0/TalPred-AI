"""Configuration dataclasses for the strategy framework.

All tunable parameters live here so nothing is hardcoded in the
construction modes, risk manager, or portfolio constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostConfig:
    """Almgren-Chriss inspired transaction cost model.

    Market impact = temporary + permanent:
      temporary  = eta * sigma * sqrt(V / ADV)   (concave in size)
      permanent  = gamma * sigma * (V / ADV)      (linear in size)
    Total slippage (bps) = (temporary + permanent) * 10_000, clamped.

    When volatility is unknown, falls back to a fixed-sigma estimate.
    """

    base_cost_bps: float = 5.0
    slippage_floor_bps: float = 1.0
    slippage_cap_bps: float = 30.0
    short_borrow_bps_annual: float = 50.0

    # Almgren-Chriss coefficients
    eta_temporary: float = 0.6
    gamma_permanent: float = 0.1
    default_daily_vol: float = 0.02

    # Partial fill modeling: if position_value > max_participation * ADV,
    # the excess is not filled. This prevents unrealistic fills in illiquid names.
    max_participation_rate: float = 0.05

    def slippage_bps(
        self,
        position_value: float,
        adv: float,
        daily_vol: float | None = None,
    ) -> float:
        """Almgren-Chriss market impact in basis points.

        Args:
            position_value: Dollar value of the trade.
            adv: 30-day average daily dollar volume.
            daily_vol: Stock's daily return volatility (optional).
        """
        if adv <= 0:
            return self.slippage_cap_bps

        sigma = daily_vol if daily_vol and daily_vol > 0 else self.default_daily_vol
        participation = position_value / adv

        temporary = self.eta_temporary * sigma * (participation ** 0.5)
        permanent = self.gamma_permanent * sigma * participation
        raw_bps = (temporary + permanent) * 10_000

        return max(self.slippage_floor_bps, min(raw_bps, self.slippage_cap_bps))

    def daily_borrow_cost(self, notional: float) -> float:
        """Daily short borrow cost for a given notional."""
        return notional * (self.short_borrow_bps_annual / 10_000) / 252


@dataclass
class MomentumConfig:
    min_confidence: float = 0.60
    top_n: int = 20
    alignment_boost: float = 1.2
    alignment_penalty: float = 0.7


@dataclass
class SectorRotationConfig:
    min_confidence: float = 0.55
    top_sectors: int = 4
    bottom_sectors: int = 3
    picks_per_sector: int = 3


@dataclass
class MeanReversionConfig:
    min_confidence: float = 0.55
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    reversal_threshold_pct: float = 2.0
    max_picks: int = 10


@dataclass
class RiskConfig:
    max_position_weight: float = 0.05
    max_sector_weight: float = 0.20
    max_gross_long: float = 0.60
    max_gross_short: float = 0.40
    vol_target_annual: float = 0.10
    vol_lookback_days: int = 20
    vol_scale_cap: float = 1.5
    adv_participation_limit: float = 0.01
    min_weight_threshold: float = 0.005
    vix_crisis: float = 30.0
    vix_elevated: float = 25.0
    vix_above_avg: float = 20.0
    drawdown_lookback_days: int = 20
    drawdown_reduce_threshold: float = -0.05
    drawdown_reduce_scale: float = 0.25
    drawdown_halt_threshold: float = -0.10
    # EWMA volatility targeting (replaces simple rolling when enabled)
    use_ewma_vol: bool = True
    ewma_halflife_days: int = 10
    # Factor exposure constraints (requires StatisticalFactorModel).
    # Opt-in: set to True to build PCA factor model and enforce per-factor
    # exposure limits. Auto-enabled when set_factor_model() is called at
    # runtime. Kept False by default to avoid unnecessary DB queries on
    # systems that don't need factor risk decomposition.
    max_factor_exposure: float = 2.0
    factor_constraint_enabled: bool = False


@dataclass
class RebalanceConfig:
    stride_days: int = 5


@dataclass
class MomentumReversalConfig:
    min_confidence: float = 0.52
    top_n: int = 20
    reversal_weight: float = 0.30
    reversal_lookback_col: str = "rolling_return_5d"
    rsi_col: str = "rsi_14"
    rsi_boost_oversold: float = 30.0
    rsi_boost_overbought: float = 70.0
    dynamic_regime_weight: bool = True
    # Reversal weight schedule keyed by VIX level:
    #   VIX >= vix_crisis  → weight_crisis  (momentum dominates)
    #   VIX >= vix_elevated → weight_elevated
    #   VIX >= vix_normal   → weight_normal  (balanced)
    #   VIX < vix_normal    → weight_calm    (reversal dominates)
    vix_crisis: float = 30.0
    vix_elevated: float = 25.0
    vix_normal: float = 18.0
    weight_crisis: float = 0.10
    weight_elevated: float = 0.20
    weight_normal: float = 0.30
    weight_calm: float = 0.45


@dataclass
class StrategyFrameworkConfig:
    """Top-level config aggregating all sub-configs."""

    cost: CostConfig = field(default_factory=CostConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    sector_rotation: SectorRotationConfig = field(default_factory=SectorRotationConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    momentum_reversal: MomentumReversalConfig = field(default_factory=MomentumReversalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    starting_capital: float = 100_000.0
    beta_neutral: bool = True
    turnover_penalty: float = 0.15
