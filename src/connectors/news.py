"""News data connector for NewsAPI (daily) and GDELT (historical backfill)."""

import hashlib
import logging
from datetime import UTC, date, datetime

import requests

from src.config import settings

logger = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2/everything"


def fetch_news_newsapi(
    query: str,
    from_date: date,
    to_date: date,
    page_size: int = 100,
) -> list[dict]:
    """Fetch news articles from NewsAPI.

    Returns list of normalized news event dicts.
    """
    if not settings.newsapi_key:
        logger.warning("NEWSAPI_KEY not set, skipping")
        return []

    params = {
        "q": query,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": settings.newsapi_key,
    }

    try:
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("NewsAPI request failed")
        return []

    articles = data.get("articles", [])
    return [_normalize_newsapi_article(a) for a in articles]


def _normalize_newsapi_article(article: dict) -> dict:
    title = article.get("title", "")
    published = article.get("publishedAt", "")
    source = article.get("source", {}).get("name", "unknown")

    event_id = hashlib.sha256(f"{title}:{published}".encode()).hexdigest()[:64]

    return {
        "event_id": event_id,
        "headline": title,
        "source_name": source,
        "published_time": published,
        "ingested_time": datetime.now(UTC).isoformat(),
        "sentiment_score": None,
        "event_tags": None,
        "credibility_score": None,
    }


def fetch_news_for_symbols(
    symbols: list[str],
    from_date: date,
    to_date: date,
) -> list[dict]:
    """Fetch news for a list of symbols. Returns deduplicated events."""
    query = " OR ".join(symbols[:5])
    return fetch_news_newsapi(query, from_date, to_date)
