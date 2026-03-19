"""MomentumLongShort portfolio construction mode.

Confidence-weighted long/short with momentum alignment filter.
Boosts score when model direction agrees with multi-timeframe momentum;
penalises when they disagree.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.strategies.base import BaseStrategy, Signal
from src.strategies.config import MomentumConfig


class MomentumLongShort(BaseStrategy):
    name = "momentum_long_short"

    def __init__(self, config: MomentumConfig | None = None) -> None:
        self.cfg = config or MomentumConfig()

    def generate_signals(
        self,
        predictions_df: pd.DataFrame,
        features_df: pd.DataFrame,
        market_df: pd.DataFrame,
        target_date: date,
    ) -> list[Signal]:
        preds = predictions_df[predictions_df["confidence"] >= self.cfg.min_confidence].copy()
        if preds.empty:
            return []

        merged = preds.merge(
            features_df[["symbol", "momentum_5d", "momentum_20d"]].drop_duplicates("symbol"),
            on="symbol",
            how="left",
        )

        signals: list[Signal] = []
        for _, row in merged.iterrows():
            direction = "long" if row["direction"] == "up" else "short"
            mom5 = row.get("momentum_5d")
            mom20 = row.get("momentum_20d")

            alignment = self._alignment_factor(direction, mom5, mom20)
            raw_score = float(row["confidence"]) * alignment
            if direction == "short":
                raw_score = -raw_score

            signals.append(
                Signal(
                    symbol=row["symbol"],
                    direction=direction,
                    raw_score=raw_score,
                    strategy_name=self.name,
                    metadata={
                        "confidence": float(row["confidence"]),
                        "momentum_5d": float(mom5) if pd.notna(mom5) else None,
                        "momentum_20d": float(mom20) if pd.notna(mom20) else None,
                        "alignment": alignment,
                    },
                )
            )

        signals.sort(key=lambda s: abs(s.raw_score), reverse=True)
        return signals[: self.cfg.top_n]

    def _alignment_factor(
        self, direction: str, mom5: float | None, mom20: float | None
    ) -> float:
        if mom5 is None or pd.isna(mom5):
            return 1.0

        mom5_agrees = (direction == "long" and mom5 > 0) or (
            direction == "short" and mom5 < 0
        )
        mom20_agrees = True
        if mom20 is not None and not pd.isna(mom20):
            mom20_agrees = (direction == "long" and mom20 > 0) or (
                direction == "short" and mom20 < 0
            )

        if mom5_agrees and mom20_agrees:
            return self.cfg.alignment_boost
        if not mom5_agrees:
            return self.cfg.alignment_penalty
        return 1.0
