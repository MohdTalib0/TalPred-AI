"""Source credibility scoring for news events.

Maps news sources to reliability scores based on editorial standards,
track record, and institutional reputation. Used to weight sentiment
signals -- high-credibility sources influence features more.

Tier system:
  Tier 1 (0.90-0.95): Wire services, major financial press with editorial standards
  Tier 2 (0.75-0.85): Established financial media, data-driven outlets
  Tier 3 (0.55-0.65): Aggregators, community-driven, mixed editorial quality
  Tier 4 (0.30-0.40): Unknown or unverified sources
"""

import logging

logger = logging.getLogger(__name__)

SOURCE_CREDIBILITY = {
    # Tier 1: Wire services & institutional financial press
    "Reuters": 0.95,
    "DowJones": 0.95,
    "Bloomberg": 0.95,
    "Associated Press": 0.93,
    "Financial Times": 0.93,
    "WSJ": 0.93,
    "Wall Street Journal": 0.93,
    "CNBC": 0.90,
    "MarketWatch": 0.88,

    # Tier 2: Established financial media & data platforms
    "Benzinga": 0.80,
    "Barrons": 0.85,
    "Investopedia": 0.80,
    "TheStreet": 0.78,
    "Fintel": 0.78,
    "Morningstar": 0.85,
    "Nasdaq": 0.82,

    # Tier 3: Aggregators, community, mixed quality
    "Yahoo": 0.65,
    "SeekingAlpha": 0.60,
    "ChartMill": 0.55,
    "Finnhub": 0.55,
    "Motley Fool": 0.58,
    "InvestorPlace": 0.55,
    "GuruFocus": 0.60,
    "TipRanks": 0.65,
    "Zacks": 0.62,

    # Tier 4: Default for unknown
    "unknown": 0.35,
}

DEFAULT_CREDIBILITY = 0.40


def get_credibility(source_name: str | None) -> float:
    """Look up credibility score for a source. Case-insensitive fuzzy match."""
    if not source_name:
        return DEFAULT_CREDIBILITY

    clean = source_name.strip()

    if clean in SOURCE_CREDIBILITY:
        return SOURCE_CREDIBILITY[clean]

    lower = clean.lower()
    for key, score in SOURCE_CREDIBILITY.items():
        if key.lower() == lower or key.lower() in lower or lower in key.lower():
            return score

    return DEFAULT_CREDIBILITY


def score_credibility_batch(source_names: list[str | None]) -> list[float]:
    """Score credibility for a batch of source names."""
    return [get_credibility(s) for s in source_names]
