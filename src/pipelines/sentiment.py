"""News sentiment pipeline.

Fetches recent headlines for each symbol from the news_events table,
scores them via OpenRouter, and stores aggregated sentiment per symbol per date.
"""

import logging
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.connectors.sentiment import score_headlines

logger = logging.getLogger(__name__)


def get_recent_headlines(
    db: Session,
    symbol: str,
    target_date: date,
    lookback_hours: int = 24,
) -> list[str]:
    """Get headlines for a symbol from the last N hours before target_date."""
    result = db.execute(text("""
        SELECT ne.headline
        FROM news_events ne
        JOIN news_symbol_mapping nsm ON ne.event_id = nsm.event_id
        WHERE nsm.symbol = :symbol
          AND ne.published_time >= :start_time
          AND ne.published_time < :end_time
        ORDER BY ne.published_time DESC
        LIMIT 50
    """), {
        "symbol": symbol,
        "start_time": f"{target_date - timedelta(hours=lookback_hours)}",
        "end_time": f"{target_date}",
    })
    return [row[0] for row in result.fetchall()]


def get_general_market_headlines(
    db: Session,
    target_date: date,
    lookback_hours: int = 24,
) -> list[str]:
    """Get recent general market headlines (not symbol-specific)."""
    result = db.execute(text("""
        SELECT headline
        FROM news_events
        WHERE published_time >= :start_time
          AND published_time < :end_time
        ORDER BY published_time DESC
        LIMIT 100
    """), {
        "start_time": f"{target_date - timedelta(hours=lookback_hours)}",
        "end_time": f"{target_date}",
    })
    return [row[0] for row in result.fetchall()]


def compute_sentiment_for_symbols(
    db: Session,
    symbols: list[str],
    target_date: date,
) -> dict[str, dict]:
    """Compute sentiment scores for all symbols.

    Returns {symbol: {"sentiment_24h": float, "sentiment_7d": float}}
    """
    all_headlines_24h = get_general_market_headlines(db, target_date, lookback_hours=24)
    all_headlines_7d = get_general_market_headlines(db, target_date, lookback_hours=168)

    market_sentiment_24h = 0.0
    market_sentiment_7d = 0.0

    if all_headlines_24h:
        scores_24h = score_headlines(all_headlines_24h)
        sentiments = [s["sentiment"] for s in scores_24h if s["confidence"] > 0.3]
        market_sentiment_24h = sum(sentiments) / len(sentiments) if sentiments else 0.0

    if all_headlines_7d:
        scores_7d = score_headlines(all_headlines_7d)
        sentiments = [s["sentiment"] for s in scores_7d if s["confidence"] > 0.3]
        market_sentiment_7d = sum(sentiments) / len(sentiments) if sentiments else 0.0

    result = {}
    for symbol in symbols:
        sym_headlines = get_recent_headlines(db, symbol, target_date, lookback_hours=24)

        if sym_headlines:
            sym_scores = score_headlines(sym_headlines)
            sym_sentiments = [s["sentiment"] for s in sym_scores if s["confidence"] > 0.3]
            sentiment_24h = sum(sym_sentiments) / len(sym_sentiments) if sym_sentiments else market_sentiment_24h
        else:
            sentiment_24h = market_sentiment_24h

        result[symbol] = {
            "sentiment_24h": sentiment_24h,
            "sentiment_7d": market_sentiment_7d,
        }

    logger.info(
        f"Sentiment scored for {len(symbols)} symbols, "
        f"market 24h={market_sentiment_24h:.3f}, 7d={market_sentiment_7d:.3f}"
    )
    return result
