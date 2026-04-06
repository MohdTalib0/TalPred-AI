"""SectorRotation portfolio construction mode.

Sector selection is driven by sector-level features (not model confidence)
to avoid double-counting the alpha source.  Stock selection within chosen
sectors uses model confidence.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, Signal
from src.strategies.config import SectorRotationConfig


class SectorRotation(BaseStrategy):
    name = "sector_rotation"

    def __init__(self, config: SectorRotationConfig | None = None) -> None:
        self.cfg = config or SectorRotationConfig()

    def generate_signals(
        self,
        predictions_df: pd.DataFrame,
        features_df: pd.DataFrame,
        market_df: pd.DataFrame,
        target_date: date,
    ) -> list[Signal]:
        feat = features_df[
            ["symbol", "sector_return_5d", "sector_momentum_rank", "sector_return_1d"]
        ].drop_duplicates("symbol")

        if "sector" not in predictions_df.columns:
            merged = predictions_df.merge(
                features_df[["symbol", "sector", "sector_momentum_rank"]].drop_duplicates("symbol"),
                on="symbol",
                how="left",
            )
            if "sector" not in merged.columns:
                return []
        else:
            merged = predictions_df.copy()

        merged = merged.merge(feat, on="symbol", how="left", suffixes=("", "_feat"))

        sector_scores = self._rank_sectors(merged)
        if not sector_scores:
            return []

        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        long_sectors = {s for s, _ in sorted_sectors[: self.cfg.top_sectors]}
        short_sectors = {s for s, _ in sorted_sectors[-self.cfg.bottom_sectors :]}

        signals: list[Signal] = []

        for sector_set, direction in [(long_sectors, "long"), (short_sectors, "short")]:
            sector_df = merged[merged["sector"].isin(sector_set)]
            if direction == "long":
                sector_df = sector_df[sector_df["confidence"] >= self.cfg.min_confidence]
            else:
                sector_df = sector_df[sector_df["confidence"] >= self.cfg.min_confidence]

            for sector in sector_set:
                sec_picks = sector_df[sector_df["sector"] == sector].nlargest(
                    self.cfg.picks_per_sector, "confidence"
                )
                for _, row in sec_picks.iterrows():
                    score = float(row["confidence"]) * (
                        sector_scores.get(sector, 0.5) if direction == "long"
                        else (1 - sector_scores.get(sector, 0.5))
                    )
                    if direction == "short":
                        score = -score

                    signals.append(
                        Signal(
                            symbol=row["symbol"],
                            direction=direction,
                            raw_score=score,
                            strategy_name=self.name,
                            metadata={
                                "sector": sector,
                                "sector_score": sector_scores.get(sector),
                                "confidence": float(row["confidence"]),
                            },
                        )
                    )

        return signals

    @staticmethod
    def _rank_sectors(merged: pd.DataFrame) -> dict[str, float]:
        """Rank sectors using sector-level features only (not model confidence)."""
        if "sector" not in merged.columns:
            return {}

        agg = merged.groupby("sector").agg(
            ret5d=("sector_return_5d", "first"),
            mom_rank=("sector_momentum_rank", "mean"),
        )
        agg = agg.dropna(how="all")
        if agg.empty:
            return {}

        for col in ["ret5d", "mom_rank"]:
            if agg[col].std() > 0:
                agg[f"{col}_z"] = (agg[col] - agg[col].mean()) / agg[col].std()
            else:
                agg[f"{col}_z"] = 0.0

        agg["composite"] = 0.6 * agg.get("ret5d_z", 0) + 0.4 * agg.get("mom_rank_z", 0)

        mi, mx = agg["composite"].min(), agg["composite"].max()
        if mx - mi > 0:
            agg["score"] = (agg["composite"] - mi) / (mx - mi)
        else:
            agg["score"] = 0.5

        return agg["score"].to_dict()
