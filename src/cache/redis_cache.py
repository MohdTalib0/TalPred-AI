"""Redis prediction cache (BE-302).

Cache schema:
  Key:   pred:{symbol}:{target_date}:{model_version}
  Value: JSON-serialized prediction payload
  TTL:   86400 seconds (24 hours) by default

Provides cache-first read with DB fallback pattern for the prediction API.
"""

import json
import logging
from datetime import date

import redis

from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_TTL = 86400


def get_redis_client() -> redis.Redis | None:
    """Create a Redis client from settings. Returns None if unavailable."""
    if not settings.redis_url:
        logger.warning("REDIS_URL not configured")
        return None
    try:
        client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        client.ping()
        return client
    except Exception:
        logger.warning("Redis connection failed, cache disabled")
        return None


def _cache_key(symbol: str, target_date: date, model_version: str) -> str:
    return f"pred:{symbol}:{target_date.isoformat()}:{model_version}"


def cache_prediction(
    r: redis.Redis | None,
    prediction: dict,
    ttl: int = DEFAULT_TTL,
) -> bool:
    """Write a prediction to Redis cache. Returns True on success."""
    if r is None:
        return False

    try:
        key = _cache_key(
            prediction["symbol"],
            prediction["target_date"] if isinstance(prediction["target_date"], date)
            else date.fromisoformat(str(prediction["target_date"])),
            prediction["model_version"],
        )

        serializable = {}
        for k, v in prediction.items():
            if isinstance(v, date):
                serializable[k] = v.isoformat()
            else:
                serializable[k] = v

        r.setex(key, ttl, json.dumps(serializable))
        return True
    except Exception:
        logger.warning(f"Cache write failed for {prediction.get('symbol')}")
        return False


def get_cached_prediction(
    r: redis.Redis | None,
    symbol: str,
    target_date: date,
    model_version: str,
) -> dict | None:
    """Read a prediction from Redis cache. Returns None on miss."""
    if r is None:
        return None

    try:
        key = _cache_key(symbol, target_date, model_version)
        data = r.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception:
        logger.warning(f"Cache read failed for {symbol}")
        return None


def cache_batch_predictions(
    r: redis.Redis | None,
    predictions: list[dict],
    ttl: int = DEFAULT_TTL,
) -> int:
    """Write multiple predictions to cache. Returns count of successful writes."""
    if r is None:
        return 0

    cached = 0
    try:
        pipe = r.pipeline()
        for pred in predictions:
            key = _cache_key(
                pred["symbol"],
                pred["target_date"] if isinstance(pred["target_date"], date)
                else date.fromisoformat(str(pred["target_date"])),
                pred["model_version"],
            )
            serializable = {}
            for k, v in pred.items():
                serializable[k] = v.isoformat() if isinstance(v, date) else v
            pipe.setex(key, ttl, json.dumps(serializable))
            cached += 1

        pipe.execute()
        logger.info(f"Cached {cached} predictions in Redis")
    except Exception:
        logger.warning("Batch cache write failed")
        cached = 0

    return cached


def invalidate_symbol_cache(r: redis.Redis | None, symbol: str) -> int:
    """Delete all cached predictions for a symbol. Returns count deleted."""
    if r is None:
        return 0

    try:
        pattern = f"pred:{symbol}:*"
        keys = list(r.scan_iter(pattern, count=100))
        if keys:
            r.delete(*keys)
        return len(keys)
    except Exception:
        logger.warning(f"Cache invalidation failed for {symbol}")
        return 0


def get_cache_stats(r: redis.Redis | None) -> dict:
    """Get basic cache statistics."""
    if r is None:
        return {"available": False}

    try:
        info = r.info("memory")
        key_count = r.dbsize()
        return {
            "available": True,
            "keys": key_count,
            "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
        }
    except Exception:
        return {"available": False}
