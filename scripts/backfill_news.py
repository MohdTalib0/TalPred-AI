"""Backfill news data and score sentiment.

Steps:
1. Ingest news for all symbols using NewsAPI (last 30 days)
2. Score headlines via OpenRouter LLM
3. Update news_events.sentiment_score

Usage:
  python -m scripts.backfill_news
  python -m scripts.backfill_news --days 7
"""

import argparse
import logging
import time
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from src.connectors.sentiment import score_headlines  # noqa: E402
from src.db import SessionLocal  # noqa: E402
from src.models.schema import NewsEvent, Symbol  # noqa: E402
from src.pipelines.ingest_news import ingest_news  # noqa: E402

from sqlalchemy import text  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("backfill_news")


def main():
    parser = argparse.ArgumentParser(description="Backfill news + sentiment")
    parser.add_argument("--days", type=int, default=29, help="Days to look back (NewsAPI free = 30 max)")
    parser.add_argument("--score-batch", type=int, default=50, help="Headlines per LLM batch")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for sentiment scoring")
    parser.add_argument("--symbol-limit", type=int, default=0, help="Limit active symbols (0 = all)")
    parser.add_argument(
        "--score-recent-days",
        type=int,
        default=0,
        help="Only score unscored headlines published in last N days (0 = all unscored)",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion, only score")
    args = parser.parse_args()

    db = SessionLocal()
    t0 = time.time()

    symbols = [
        row.symbol for row in
        db.query(Symbol).filter(Symbol.is_active.is_(True)).all()
    ]
    if args.symbol_limit and args.symbol_limit > 0:
        symbols = symbols[: args.symbol_limit]
    logger.info(f"Universe: {len(symbols)} active symbols")

    to_date = date.today()
    from_date = to_date - timedelta(days=args.days)

    # Step 1: Ingest news
    if not args.skip_ingest:
        logger.info(f"Ingesting news from {from_date} to {to_date}...")

        batch_size = 10
        total_result = {"inserted": 0, "skipped": 0, "quarantined": 0, "mappings": 0}

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbols) + batch_size - 1) // batch_size

            logger.info(f"  News batch {batch_num}/{total_batches}: {batch[0]}..{batch[-1]}")
            try:
                result = ingest_news(db, batch, from_date, to_date)
                for k in total_result:
                    total_result[k] += result.get(k, 0)
            except Exception:
                logger.exception(f"  Batch {batch_num} failed")

            time.sleep(1)

        logger.info(f"News ingestion complete: {total_result}")

    # Step 2: Score unscored headlines using parallel processing
    logger.info("Scoring unscored headlines (parallel, 8 workers)...")
    unscored_q = db.query(NewsEvent).filter(NewsEvent.sentiment_score.is_(None))
    if args.score_recent_days and args.score_recent_days > 0:
        cutoff = datetime.now(UTC) - timedelta(days=args.score_recent_days)
        unscored_q = unscored_q.filter(NewsEvent.published_time >= cutoff)
    unscored = unscored_q.all()
    logger.info(f"  {len(unscored)} unscored headlines found")

    mega_batch = 400
    scored_count = 0

    for start in range(0, len(unscored), mega_batch):
        chunk = unscored[start : start + mega_batch]
        headlines = [e.headline for e in chunk]

        results = score_headlines(headlines, batch_size=args.score_batch, workers=args.workers)

        count = 0
        for event, result in zip(chunk, results):
            sentiment = result.get("sentiment", 0.0)
            confidence = result.get("confidence", 0.0)
            if confidence > 0.2:
                event.sentiment_score = sentiment
                count += 1

        db.commit()
        scored_count += count
        logger.info(
            f"  Progress: {scored_count}/{len(unscored)} scored "
            f"(chunk {start//mega_batch + 1}/{(len(unscored) + mega_batch - 1)//mega_batch})"
        )

    elapsed = time.time() - t0
    logger.info(f"Backfill complete: {scored_count} headlines scored in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    db.close()


if __name__ == "__main__":
    main()
