"""News sentiment analysis via OpenRouter LLM API.

Sends batched headlines to an LLM for sentiment scoring (-1 to +1).
Supports parallel batch processing for high throughput.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import requests

from src.config import settings

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are a financial sentiment analyst. For each headline, return a JSON array of objects with:
- "index": the headline number (starting from 0)
- "sentiment": a float from -1.0 (very bearish) to +1.0 (very bullish), 0.0 is neutral
- "confidence": a float from 0.0 to 1.0 indicating how confident you are

Only return the JSON array, no other text. Example:
[{"index": 0, "sentiment": 0.7, "confidence": 0.9}, {"index": 1, "sentiment": -0.3, "confidence": 0.8}]"""


def score_headlines(
    headlines: list[str],
    model: str | None = None,
    batch_size: int = 50,
    workers: int = 8,
) -> list[dict]:
    """Score headlines with parallel batch processing.

    Splits headlines into batches and scores them concurrently via ThreadPoolExecutor.
    With workers=8, batch_size=50: ~10-50x faster than sequential.
    """
    if model is None:
        model = settings.openrouter_model_name

    if not settings.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY not set, returning neutral scores")
        return [
            {"headline": h, "sentiment": 0.0, "confidence": 0.0, "model": "none", "scored_at": datetime.now(UTC)}
            for h in headlines
        ]

    batches = [headlines[i : i + batch_size] for i in range(0, len(headlines), batch_size)]

    if len(batches) <= 1:
        return _score_batch(batches[0], model) if batches else []

    all_results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_score_batch, batch, model): idx
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


def _score_batch(headlines: list[str], model: str) -> list[dict]:
    """Score a single batch of headlines via OpenRouter."""
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

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://talpred-ai.local",
        "X-Title": "TalPred-AI Sentiment",
    }

    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
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

    except Exception:
        logger.exception("OpenRouter sentiment request failed")
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
    """Score headlines for a symbol and return aggregated sentiment.

    Returns dict with: symbol, sentiment_24h (avg), num_headlines, scored_at
    """
    if model is None:
        model = settings.openrouter_model_name

    if not headlines:
        return {"symbol": symbol, "sentiment_24h": None, "num_headlines": 0}

    results = score_headlines(headlines, model)
    sentiments = [r["sentiment"] for r in results if r["confidence"] > 0.3]

    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

    return {
        "symbol": symbol,
        "sentiment_24h": avg_sentiment,
        "num_headlines": len(headlines),
        "scored_at": datetime.now(UTC),
    }
