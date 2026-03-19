"""MeanReversion portfolio construction mode.

Trades only when model direction aligns with extreme technical readings
(oversold/overbought RSI + reversal magnitude).  Produces fewer signals
than the other modes but with higher per-trade conviction.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.strategies.base import BaseStrategy, Signal
from src.strategies.config import MeanReversionConfig


class MeanReversion(BaseStrategy):
    name = "mean_reversion"

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.cfg = config or MeanReversionConfig()

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

        feat_cols = ["symbol", "rsi_14", "short_term_reversal"]
        avail = [c for c in feat_cols if c in features_df.columns]
        merged = preds.merge(
            features_df[avail].drop_duplicates("symbol"),
            on="symbol",
            how="left",
        )

        signals: list[Signal] = []
        for _, row in merged.iterrows():
            rsi = row.get("rsi_14")
            reversal = row.get("short_term_reversal")

            if rsi is None or pd.isna(rsi):
                continue
            if reversal is None or pd.isna(reversal):
                continue

            direction = row["direction"]
            reversal_pct = reversal * 100 if abs(reversal) < 1 else reversal

            is_long = (
                direction == "up"
                and rsi < self.cfg.rsi_oversold
                and reversal_pct < -self.cfg.reversal_threshold_pct
            )
            is_short = (
                direction == "down"
                and rsi > self.cfg.rsi_overbought
                and reversal_pct > self.cfg.reversal_threshold_pct
            )

            if not is_long and not is_short:
                continue

            extremity = abs(rsi - 50) / 50
            score = extremity * float(row["confidence"])
            sig_direction = "long" if is_long else "short"
            if sig_direction == "short":
                score = -score

            signals.append(
                Signal(
                    symbol=row["symbol"],
                    direction=sig_direction,
                    raw_score=score,
                    strategy_name=self.name,
                    metadata={
                        "rsi": float(rsi),
                        "reversal_pct": reversal_pct,
                        "confidence": float(row["confidence"]),
                    },
                )
            )

        signals.sort(key=lambda s: abs(s.raw_score), reverse=True)
        return signals[: self.cfg.max_picks]
