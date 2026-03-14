"""News sentiment analysis via OpenRouter LLM API.

Sends batched headlines to an LLM for sentiment scoring (-1 to +1).
Uses HTTP session reuse, retry logic, parallel batch processing,
and headline cache for deduplication.
"""

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import settings

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are a financial sentiment analyst. For each headline, return a JSON array of objects with:
- "index": the headline number (starting from 0)
- "sentiment": a float from -1.0 (very bearish) to +1.0 (very bullish), 0.0 is neutral
- "confidence": a float from 0.0 to 1.0 indicating how confident you are

Only return the JSON array, no other text. Example:
[{"index": 0, "sentiment": 0.7, "confidence": 0.9}, {"index": 1, "sentiment": -0.3, "confidence": 0.8}]"""


def _build_session() -> requests.Session:
    """Create a session with connection pooling and automatic retries."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )
    session.mount("https://", adapter)
    session.headers.update({
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://talpred-ai.local",
        "X-Title": "TalPred-AI Sentiment",
    })
    return session


_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Thread-safe lazy session singleton."""
    global _session
    if _session is None:
        _session = _build_session()
    return _session


def score_headlines(
    headlines: list[str],
    model: str | None = None,
    batch_size: int = 50,
    workers: int = 8,
) -> list[dict]:
    """Score headlines with parallel batch processing.

    Uses persistent HTTP session with connection pooling and automatic retries.
    Splits headlines into batches and scores them concurrently.
    """
    if model is None:
        model = settings.openrouter_model_name

    if not settings.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY not set, returning neutral scores")
        return [
            {"headline": h, "sentiment": 0.0, "confidence": 0.0, "model": "none", "scored_at": datetime.now(UTC)}
            for h in headlines
        ]

    session = _get_session()
    batches = [headlines[i : i + batch_size] for i in range(0, len(headlines), batch_size)]

    if len(batches) <= 1:
        return _score_batch(batches[0], model, session) if batches else []

    all_results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_score_batch, batch, model, session): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_results[idx] = future.result()
            except Exception:
                logger.exception(f"Batch {idx} scoring failed")
                now = datetime.now(UTC)
                all_results[idx] = [
                    {"headline": h, "sentiment": 0.0, "confidence": 0.0, "model": model, "scored_at": now}
                    for h in batches[idx]
                ]

    flat = []
    for batch_result in all_results:
        if batch_result:
            flat.extend(batch_result)
    return flat


def _score_batch(headlines: list[str], model: str, session: requests.Session) -> list[dict]:
    """Score a single batch via OpenRouter with retry on transient failures."""
    numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    user_msg = f"Score the sentiment of these financial headlines:\n\n{numbered}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    last_exc = None
    for attempt in range(3):
        try:
            resp = session.post(OPENROUTER_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"].strip()
            scores = _parse_scores(content)

            now = datetime.now(UTC)
            results = []
            for i, headline in enumerate(headlines):
                score_data = scores.get(i, {"sentiment": 0.0, "confidence": 0.0})
                results.append({
                    "headline": headline,
                    "sentiment": float(score_data.get("sentiment", 0.0)),
                    "confidence": float(score_data.get("confidence", 0.0)),
                    "model": model,
                    "scored_at": now,
                })
            return results

        except Exception as e:
            last_exc = e
            if attempt < 2:
                wait = 2 ** attempt
                logger.warning(f"Batch scoring attempt {attempt + 1} failed, retrying in {wait}s: {e}")
                time.sleep(wait)

    logger.error(f"Batch scoring failed after 3 attempts: {last_exc}")
    now = datetime.now(UTC)
    return [
        {"headline": h, "sentiment": 0.0, "confidence": 0.0, "model": model, "scored_at": now}
        for h in headlines
    ]


def _parse_scores(content: str) -> dict[int, dict]:
    """Parse LLM JSON response into {index: {sentiment, confidence}} map."""
    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            return {}
        arr = json.loads(content[start:end])
        return {item["index"]: item for item in arr if "index" in item}
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Failed to parse sentiment response: {content[:200]}")
        return {}


def score_news_for_symbol(
    symbol: str,
    headlines: list[str],
    model: str | None = None,
) -> dict:
    """Score headlines for a symbol and return confidence-weighted aggregated sentiment.

    Returns dict with: symbol, sentiment_mean, sentiment_std, positive_ratio,
                       negative_ratio, news_volume, scored_at
    """
    if model is None:
        model = settings.openrouter_model_name

    if not headlines:
        return {
            "symbol": symbol,
            "sentiment_24h": None,
            "sentiment_std": None,
            "positive_ratio": None,
            "negative_ratio": None,
            "news_volume": 0,
            "num_headlines": 0,
        }

    results = score_headlines(headlines, model)
    high_conf = [r for r in results if r["confidence"] > 0.3]

    if not high_conf:
        return {
            "symbol": symbol,
            "sentiment_24h": 0.0,
            "sentiment_std": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "news_volume": len(headlines),
            "num_headlines": len(headlines),
            "scored_at": datetime.now(UTC),
        }

    weighted_sum = sum(r["sentiment"] * r["confidence"] for r in high_conf)
    weight_total = sum(r["confidence"] for r in high_conf)
    avg_sentiment = weighted_sum / weight_total if weight_total else 0.0

    sentiments = [r["sentiment"] for r in high_conf]
    n = len(sentiments)
    mean = sum(sentiments) / n
    std = (sum((s - mean) ** 2 for s in sentiments) / n) ** 0.5 if n > 1 else 0.0
    positive_ratio = sum(1 for s in sentiments if s > 0.1) / n
    negative_ratio = sum(1 for s in sentiments if s < -0.1) / n

    return {
        "symbol": symbol,
        "sentiment_24h": avg_sentiment,
        "sentiment_std": std,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
        "news_volume": len(headlines),
        "num_headlines": len(headlines),
        "scored_at": datetime.now(UTC),
    }


def _headline_hash(headline: str) -> str:
    return hashlib.sha256(headline.strip().lower().encode()).hexdigest()[:64]


def score_with_cache(
    headlines: list[str],
    db_session,
    model: str | None = None,
    batch_size: int = 50,
    workers: int = 8,
) -> list[dict]:
    """Score headlines with a DB-backed cache to avoid re-scoring duplicates.

    Checks headline_sentiment_cache first, only sends unseen headlines to the LLM.
    Typically reduces LLM calls by 30-80%.
    """
    from sqlalchemy import text as sa_text

    if model is None:
        model = settings.openrouter_model_name

    hashes = {_headline_hash(h): h for h in headlines}
    hash_list = list(hashes.keys())

    cached_rows = db_session.execute(
        sa_text("SELECT headline_hash, sentiment, confidence FROM headline_sentiment_cache WHERE headline_hash = ANY(:hashes)"),
        {"hashes": hash_list},
    ).fetchall()
    cached = {r[0]: {"sentiment": r[1], "confidence": r[2]} for r in cached_rows}

    results = []
    uncached_headlines = []
    uncached_indices = []

    for i, h in enumerate(headlines):
        hh = _headline_hash(h)
        if hh in cached:
            results.append({
                "headline": h,
                "sentiment": cached[hh]["sentiment"],
                "confidence": cached[hh]["confidence"],
                "model": model,
                "scored_at": datetime.now(UTC),
                "source": "cache",
            })
        else:
            results.append(None)
            uncached_headlines.append(h)
            uncached_indices.append(i)

    if uncached_headlines:
        logger.info(f"Cache hit: {len(headlines) - len(uncached_headlines)}/{len(headlines)}, scoring {len(uncached_headlines)} new")
        scored = score_headlines(uncached_headlines, model, batch_size, workers)

        for idx, score_result in zip(uncached_indices, scored):
            score_result["source"] = "llm"
            results[idx] = score_result

            hh = _headline_hash(score_result["headline"])
            try:
                db_session.execute(
                    sa_text("""
                        INSERT INTO headline_sentiment_cache (headline_hash, headline, sentiment, confidence, model, scored_at)
                        VALUES (:hh, :headline, :sentiment, :confidence, :model, :scored_at)
                        ON CONFLICT (headline_hash) DO NOTHING
                    """),
                    {
                        "hh": hh,
                        "headline": score_result["headline"][:500],
                        "sentiment": score_result["sentiment"],
                        "confidence": score_result["confidence"],
                        "model": model,
                        "scored_at": score_result["scored_at"],
                    },
                )
            except Exception:
                logger.debug(f"Cache write failed for {hh[:8]}")

        db_session.commit()
    else:
        logger.info(f"100% cache hit: all {len(headlines)} headlines already scored")

    return results
