"""Pluggable portfolio construction framework.

Provides three construction modes (MomentumLongShort, SectorRotation,
MeanReversion) that operate on the same XGBoost prediction signal, plus
a shared RiskManager and PortfolioConstructor.
"""

from src.strategies.base import BaseStrategy, Signal
from src.strategies.config import (
    CostConfig,
    MeanReversionConfig,
    MomentumConfig,
    RebalanceConfig,
    RiskConfig,
    SectorRotationConfig,
    StrategyFrameworkConfig,
)
from src.strategies.mean_reversion import MeanReversion
from src.strategies.momentum_long_short import MomentumLongShort
from src.strategies.portfolio import PortfolioConstructor
from src.strategies.risk_manager import RiskManager
from src.strategies.sector_rotation import SectorRotation

__all__ = [
    "BaseStrategy",
    "Signal",
    "MomentumLongShort",
    "SectorRotation",
    "MeanReversion",
    "RiskManager",
    "PortfolioConstructor",
    "StrategyFrameworkConfig",
    "MomentumConfig",
    "SectorRotationConfig",
    "MeanReversionConfig",
    "RiskConfig",
    "CostConfig",
    "RebalanceConfig",
]
