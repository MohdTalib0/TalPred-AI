"""News data connector using Finnhub (free tier company news)."""

import hashlib
import logging
import time
from datetime import UTC, date, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import settings

logger = logging.getLogger(__name__)

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_GENERAL_URL = "https://finnhub.io/api/v1/news"

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
    ),
)


def fetch_news_finnhub(
    symbol: str,
    from_date: date,
    to_date: date,
) -> list[dict]:
    """Fetch company news from Finnhub for a single symbol."""
    if not settings.finnhub_api_key:
        logger.warning("FINNHUB_API_KEY not set, skipping")
        return []

    params = {
        "symbol": symbol,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": settings.finnhub_api_key,
    }

    articles = []
    for attempt in range(3):
        try:
            resp = _session.get(FINNHUB_NEWS_URL, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else min(4.0, 1.5 * (attempt + 1))
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            parsed = resp.json()
            if not isinstance(parsed, list):
                return []
            articles = parsed
            break
        except requests.RequestException:
            if attempt == 2:
                logger.warning("Finnhub news request failed for %s after retries", symbol)
                return []
            time.sleep(0.5 * (attempt + 1))

    return [_normalize_finnhub_article(a, symbol) for a in articles if a.get("headline")]


def fetch_general_news_finnhub(category: str = "general") -> list[dict]:
    """Fetch general market news from Finnhub."""
    if not settings.finnhub_api_key:
        return []

    params = {"category": category, "token": settings.finnhub_api_key}

    try:
        resp = requests.get(FINNHUB_GENERAL_URL, params=params, timeout=30)
        resp.raise_for_status()
        articles = resp.json()
        if not isinstance(articles, list):
            return []
    except Exception:
        logger.exception("Finnhub general news request failed")
        return []

    return [_normalize_finnhub_article(a) for a in articles if a.get("headline")]


def _normalize_finnhub_article(article: dict, symbol: str | None = None) -> dict:
    headline = article.get("headline", "")
    published_ts = article.get("datetime", 0)
    source = article.get("source", "unknown")
    url = article.get("url", "")

    published_str = datetime.fromtimestamp(published_ts, tz=UTC).isoformat() if published_ts else ""
    event_id = hashlib.sha256(f"{headline}:{published_str}".encode()).hexdigest()[:64]

    from src.connectors.credibility import get_credibility

    result = {
        "event_id": event_id,
        "headline": headline,
        "source_name": source,
        "published_time": published_str,
        "ingested_time": datetime.now(UTC).isoformat(),
        "sentiment_score": None,
        "event_tags": {"url": url, "category": article.get("category", "")},
        "credibility_score": get_credibility(source),
    }
    if symbol:
        result["_matched_symbols"] = [symbol]
    return result


def fetch_news_for_symbols(
    symbols: list[str],
    from_date: date,
    to_date: date,
    batch_size: int = 10,
) -> list[dict]:
    """Fetch news for a list of symbols via Finnhub. Returns deduplicated events with symbol tags."""
    seen_ids = set()
    all_articles = []

    failed = 0
    for i, symbol in enumerate(symbols):
        articles = fetch_news_finnhub(symbol, from_date, to_date)
        if not articles:
            failed += 1

        for article in articles:
            if "_matched_symbols" not in article:
                article["_matched_symbols"] = []
            if symbol not in article["_matched_symbols"]:
                article["_matched_symbols"].append(symbol)

            if article["event_id"] not in seen_ids:
                seen_ids.add(article["event_id"])
                all_articles.append(article)

        # Free-tier safe pacing to reduce 429 bursts.
        if (i + 1) % 20 == 0:
            time.sleep(1.0)
        else:
            time.sleep(0.1)

    logger.info(
        "Fetched %s unique articles for %s symbols (empty_or_failed=%s)",
        len(all_articles),
        len(symbols),
        failed,
    )
    return all_articles
