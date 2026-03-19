"""Base abstractions for portfolio construction modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class Signal:
    """A single directional signal produced by a construction mode."""

    symbol: str
    direction: str  # "long" | "short"
    raw_score: float  # [-1.0, 1.0] conviction strength
    strategy_name: str
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base for all portfolio construction modes.

    Subclasses receive the same model predictions + feature data and return
    a list of ``Signal`` objects representing desired positions.
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(
        self,
        predictions_df: pd.DataFrame,
        features_df: pd.DataFrame,
        market_df: pd.DataFrame,
        target_date: date,
    ) -> list[Signal]:
        """Produce signals for a single trading day.

        Args:
            predictions_df: rows for *target_date* with at minimum
                symbol, direction, probability_up, confidence, model_version.
            features_df: latest features_snapshot rows keyed by symbol.
            market_df: recent market_bars_daily rows (for ADV, prices, etc.).
            target_date: the date being traded.

        Returns:
            List of Signal objects (may be empty if no trades qualify).
        """
        ...
