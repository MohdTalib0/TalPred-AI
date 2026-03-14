"""Backfill credibility scores on all news_events.

Maps source_name to a credibility score (0-1) based on source tier.
Uses direct psycopg2 for speed over Supabase.

Usage:
  python -m scripts.backfill_credibility
"""

import logging
import time

from dotenv import load_dotenv

load_dotenv()

import psycopg2  # noqa: E402

from src.config import settings  # noqa: E402
from src.connectors.credibility import DEFAULT_CREDIBILITY, SOURCE_CREDIBILITY  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)
logger = logging.getLogger("backfill_credibility")


def main():
    t0 = time.time()
    dsn = settings.database_url.replace("+psycopg2", "")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '120s'")
    conn.commit()

    total = 0
    batch_size = 500

    # Get distinct sources
    cur.execute("SELECT DISTINCT source_name FROM news_events")
    sources = [r[0] for r in cur.fetchall()]
    logger.info(f"Found {len(sources)} distinct sources")

    for source in sources:
        score = SOURCE_CREDIBILITY.get(source, DEFAULT_CREDIBILITY)
        # Check for fuzzy matches
        if source not in SOURCE_CREDIBILITY:
            for key, val in SOURCE_CREDIBILITY.items():
                if key.lower() in (source or "").lower() or (source or "").lower() in key.lower():
                    score = val
                    break

        while True:
            cur.execute(
                "UPDATE news_events SET credibility_score = %s "
                "WHERE event_id IN ("
                "  SELECT event_id FROM news_events "
                "  WHERE source_name = %s AND credibility_score IS DISTINCT FROM %s "
                "  LIMIT %s"
                ")",
                (score, source, score, batch_size),
            )
            conn.commit()
            if cur.rowcount == 0:
                break
            total += cur.rowcount
            logger.info(f"  {str(source):20s} -> {score:.2f}  (+{cur.rowcount}, total: {total})")

    elapsed = time.time() - t0
    logger.info(f"Updates done: {total} rows in {elapsed:.1f}s")

    cur.execute(
        "SELECT source_name, credibility_score, COUNT(*) "
        "FROM news_events GROUP BY source_name, credibility_score "
        "ORDER BY COUNT(*) DESC"
    )
    rows = cur.fetchall()
    logger.info("Final distribution:")
    for r in rows:
        logger.info(f"  {str(r[0]):20s}  cred={r[1]:.2f}  count={r[2]}")

    cur.close()
    conn.close()
    logger.info(f"Done in {(time.time() - t0):.1f}s")


if __name__ == "__main__":
    main()
