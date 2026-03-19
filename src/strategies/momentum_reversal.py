"""MomentumReversal composite strategy.

Blends two orthogonal alphas:
  1. Cross-sectional momentum (model predictions) — the primary signal
  2. Short-term mean reversion — contrarian overlay

Regime-adaptive weighting (when dynamic_regime_weight=True):
  VIX >= 30  (crisis)   → reversal_weight = 0.10  (momentum dominates)
  VIX >= 25  (elevated) → reversal_weight = 0.20
  VIX >= 18  (normal)   → reversal_weight = 0.30  (balanced)
  VIX < 18   (calm)     → reversal_weight = 0.45  (reversal dominates)

Rationale backed by our own backtest data:
  - Crisis: IC = 0.151 for momentum → lean into it
  - Low-vol: IC = 0.036 for momentum → blend with reversal for stability

Implementation:
  - Momentum score: model's probability_up, rank-normalised
  - Reversal score: -1 * rank(5d_return), so recently beaten-down
    stocks score high (expecting bounce) and recently surged stocks
    score low (expecting pullback)
  - RSI boost: extra push for extreme RSI readings
  - Composite: (1 - w) * momentum + w * reversal
  - Top N composite scores → long, Bottom N → short
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, Signal
from src.strategies.config import MomentumReversalConfig

logger = logging.getLogger(__name__)


class MomentumReversal(BaseStrategy):
    """Composite alpha: momentum + short-term reversal with regime-adaptive weights."""

    name = "momentum_reversal"

    def __init__(self, config: MomentumReversalConfig | None = None) -> None:
        self.cfg = config or MomentumReversalConfig()

    def _resolve_reversal_weight(self, features_df: pd.DataFrame) -> float:
        """Pick reversal weight based on current VIX regime.

        Falls back to the static config value when dynamic weighting
        is disabled or VIX data is unavailable.
        """
        if not self.cfg.dynamic_regime_weight:
            return self.cfg.reversal_weight

        vix = None
        if "vix_level" in features_df.columns:
            vix_vals = features_df["vix_level"].dropna()
            if not vix_vals.empty:
                vix = float(vix_vals.iloc[0])

        if vix is None:
            return self.cfg.reversal_weight

        if vix >= self.cfg.vix_crisis:
            w = self.cfg.weight_crisis
        elif vix >= self.cfg.vix_elevated:
            t = (vix - self.cfg.vix_elevated) / (self.cfg.vix_crisis - self.cfg.vix_elevated)
            w = self.cfg.weight_elevated + t * (self.cfg.weight_crisis - self.cfg.weight_elevated)
        elif vix >= self.cfg.vix_normal:
            t = (vix - self.cfg.vix_normal) / (self.cfg.vix_elevated - self.cfg.vix_normal)
            w = self.cfg.weight_normal + t * (self.cfg.weight_elevated - self.cfg.weight_normal)
        else:
            w = self.cfg.weight_calm

        logger.debug("MomentumReversal: VIX=%.1f → reversal_weight=%.2f", vix, w)
        return w

    def generate_signals(
        self,
        predictions_df: pd.DataFrame,
        features_df: pd.DataFrame,
        market_df: pd.DataFrame,
        target_date: date,
    ) -> list[Signal]:
        preds = predictions_df[
            predictions_df["confidence"] >= self.cfg.min_confidence
        ].copy()
        if preds.empty:
            return []

        feat_cols = ["symbol"]
        if self.cfg.reversal_lookback_col in features_df.columns:
            feat_cols.append(self.cfg.reversal_lookback_col)
        if self.cfg.rsi_col in features_df.columns:
            feat_cols.append(self.cfg.rsi_col)

        merged = preds.merge(
            features_df[feat_cols].drop_duplicates("symbol"),
            on="symbol",
            how="left",
        )

        merged["mom_rank"] = merged["probability_up"].rank(pct=True)

        rev_col = self.cfg.reversal_lookback_col
        if rev_col in merged.columns and merged[rev_col].notna().sum() > 10:
            merged["rev_rank"] = 1.0 - merged[rev_col].rank(pct=True)
        else:
            merged["rev_rank"] = 0.5

        rsi_col = self.cfg.rsi_col
        if rsi_col in merged.columns:
            rsi = merged[rsi_col].fillna(50.0)
            oversold_boost = (rsi < self.cfg.rsi_boost_oversold).astype(float) * 0.1
            overbought_boost = (rsi > self.cfg.rsi_boost_overbought).astype(float) * 0.1
            merged["rev_rank"] = (
                merged["rev_rank"] + oversold_boost - overbought_boost
            ).clip(0, 1)

        w = self._resolve_reversal_weight(features_df)
        merged["composite"] = (1 - w) * merged["mom_rank"] + w * merged["rev_rank"]

        merged = merged.sort_values("composite", ascending=False)

        signals: list[Signal] = []

        long_picks = merged.head(self.cfg.top_n)
        for _, row in long_picks.iterrows():
            signals.append(
                Signal(
                    symbol=row["symbol"],
                    direction="long",
                    raw_score=float(row["composite"]),
                    strategy_name=self.name,
                    metadata={
                        "prob_up": float(row["probability_up"]),
                        "mom_rank": float(row["mom_rank"]),
                        "rev_rank": float(row["rev_rank"]),
                        "composite": float(row["composite"]),
                        "reversal_weight": w,
                    },
                )
            )

        short_picks = merged.tail(self.cfg.top_n)
        for _, row in short_picks.iterrows():
            signals.append(
                Signal(
                    symbol=row["symbol"],
                    direction="short",
                    raw_score=-float(1.0 - row["composite"]),
                    strategy_name=self.name,
                    metadata={
                        "prob_up": float(row["probability_up"]),
                        "mom_rank": float(row["mom_rank"]),
                        "rev_rank": float(row["rev_rank"]),
                        "composite": float(row["composite"]),
                        "reversal_weight": w,
                    },
                )
            )

        return signals
