"""News events ingestion pipeline.

Fetches news via NewsAPI, deduplicates, and upserts into news_events + news_symbol_mapping.
"""

import logging
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from src.connectors.news import fetch_news_for_symbols
from src.models.schema import NewsEvent, NewsSymbolMapping, QuarantineRecord

logger = logging.getLogger(__name__)


def validate_news_event(record: dict) -> tuple[bool, str]:
    if not record.get("headline"):
        return False, "missing headline"
    if not record.get("published_time"):
        return False, "missing published_time"
    if record.get("sentiment_score") is not None:
        score = record["sentiment_score"]
        if score < -1 or score > 1:
            return False, f"sentiment_score out of range: {score}"
    return True, ""


def ingest_news(
    db: Session,
    symbols: list[str],
    from_date: date,
    to_date: date,
) -> dict:
    """Ingest news for symbol list. Returns counts."""
    articles = fetch_news_for_symbols(symbols, from_date, to_date)

    inserted = 0
    skipped = 0
    quarantined = 0

    for article in articles:
        is_valid, reason = validate_news_event(article)
        if not is_valid:
            db.add(QuarantineRecord(
                source="news_events",
                record_data=article,
                failure_reason=reason,
            ))
            quarantined += 1
            continue

        existing = db.query(NewsEvent).filter(
            NewsEvent.event_id == article["event_id"]
        ).first()

        if existing:
            skipped += 1
            continue

        published = article["published_time"]
        if isinstance(published, str):
            try:
                published = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except ValueError:
                published = datetime.now(UTC)

        event = NewsEvent(
            event_id=article["event_id"],
            headline=article["headline"],
            source_name=article.get("source_name"),
            published_time=published,
            ingested_time=datetime.now(UTC),
            sentiment_score=article.get("sentiment_score"),
            event_tags=article.get("event_tags"),
            credibility_score=article.get("credibility_score"),
        )
        db.add(event)
        inserted += 1

    db.commit()
    logger.info(f"News: inserted={inserted}, skipped={skipped}, quarantined={quarantined}")
    return {"inserted": inserted, "skipped": skipped, "quarantined": quarantined}
