"""MeanReversion portfolio construction mode.

Trades only when model direction aligns with extreme technical readings
(oversold/overbought RSI + reversal magnitude).  Produces fewer signals
than the other modes but with higher per-trade conviction.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.strategies.base import BaseStrategy, Signal
from src.strategies.config import MeanReversionConfig

logger = logging.getLogger(__name__)


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
        n_has_rsi = 0
        n_has_reversal = 0
        n_rsi_extreme = 0
        n_direction_match = 0

        for _, row in merged.iterrows():
            rsi = row.get("rsi_14")
            reversal = row.get("short_term_reversal")

            if rsi is None or pd.isna(rsi):
                continue
            n_has_rsi += 1
            if reversal is None or pd.isna(reversal):
                continue
            n_has_reversal += 1

            direction = row["direction"]
            reversal_pct = reversal * 100 if abs(reversal) < 1 else reversal

            rsi_oversold = rsi < self.cfg.rsi_oversold
            rsi_overbought = rsi > self.cfg.rsi_overbought
            if rsi_oversold or rsi_overbought:
                n_rsi_extreme += 1

            is_long = (
                direction == "up"
                and rsi_oversold
                and reversal_pct < -self.cfg.reversal_threshold_pct
            )
            is_short = (
                direction == "down"
                and rsi_overbought
                and reversal_pct > self.cfg.reversal_threshold_pct
            )

            if is_long or is_short:
                n_direction_match += 1

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

        logger.info(
            "MeanReversion filter funnel: %d confident → %d has_rsi → "
            "%d has_reversal → %d rsi_extreme → %d direction_match → %d signals",
            len(preds), n_has_rsi, n_has_reversal, n_rsi_extreme,
            n_direction_match, len(signals),
        )

        signals.sort(key=lambda s: abs(s.raw_score), reverse=True)
        return signals[: self.cfg.max_picks]
