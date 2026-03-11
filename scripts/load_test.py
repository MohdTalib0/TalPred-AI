"""API load testing (OP-302).

Tests cache-hit path performance against SLO: p95 <= 250ms at 100 req/s burst.
Uses concurrent threads to simulate load without external dependencies.

Usage:
  python -m scripts.load_test
  python -m scripts.load_test --url http://production-host:8000 --rps 100 --duration 30
"""

import argparse
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)

TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"]


def send_request(client: httpx.Client, base_url: str, symbol: str) -> dict:
    """Send a single prediction request and measure latency."""
    start = time.perf_counter()
    try:
        resp = client.post(
            f"{base_url}/predict",
            json={"symbol": symbol, "as_of_date": "2026-03-10"},
            timeout=5.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "status": resp.status_code,
            "latency_ms": elapsed_ms,
            "error": None,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "status": 0,
            "latency_ms": elapsed_ms,
            "error": str(e),
        }


def run_health_check(base_url: str) -> bool:
    """Verify the API is up before load testing."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def run_load_test(base_url: str, rps: int, duration_seconds: int) -> dict:
    """Run the load test at target RPS for specified duration."""
    total_requests = rps * duration_seconds
    logger.info(f"Target: {rps} req/s for {duration_seconds}s = {total_requests} requests")

    results = []
    symbol_cycle = TEST_SYMBOLS * ((total_requests // len(TEST_SYMBOLS)) + 1)

    client = httpx.Client()
    interval = 1.0 / rps

    test_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=min(rps, 50)) as pool:
        futures = []
        for i in range(total_requests):
            target_time = test_start + (i * interval)
            now = time.perf_counter()
            if target_time > now:
                time.sleep(target_time - now)

            sym = symbol_cycle[i % len(symbol_cycle)]
            futures.append(pool.submit(send_request, client, base_url, sym))

        for future in as_completed(futures):
            results.append(future.result())

    client.close()
    actual_duration = time.perf_counter() - test_start

    latencies = [r["latency_ms"] for r in results]
    errors = [r for r in results if r["error"] or r["status"] >= 500]
    status_counts = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    report = {
        "total_requests": len(results),
        "actual_duration_s": round(actual_duration, 2),
        "actual_rps": round(len(results) / actual_duration, 1),
        "latency_ms": {
            "min": round(min(latencies), 2),
            "avg": round(statistics.mean(latencies), 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max(latencies), 2),
        },
        "errors": len(errors),
        "error_rate_pct": round(len(errors) / len(results) * 100, 2),
        "status_codes": status_counts,
        "slo_p95_250ms": p95 <= 250,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="API load test")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps", type=int, default=50, help="Requests per second")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    args = parser.parse_args()

    logger.info(f"Load test target: {args.url}")

    if not run_health_check(args.url):
        logger.error("Health check failed. Is the API running?")
        return

    logger.info("Health check passed. Starting load test...")
    report = run_load_test(args.url, args.rps, args.duration)

    logger.info(f"\n{'='*60}")
    logger.info(f"LOAD TEST REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"  Total requests: {report['total_requests']}")
    logger.info(f"  Duration: {report['actual_duration_s']}s")
    logger.info(f"  Actual RPS: {report['actual_rps']}")
    logger.info(f"  Errors: {report['errors']} ({report['error_rate_pct']}%)")
    logger.info(f"  Status codes: {report['status_codes']}")
    logger.info(f"\n  Latency (ms):")
    for k, v in report["latency_ms"].items():
        logger.info(f"    {k:>4}: {v:.2f}")
    logger.info(f"\n  SLO (p95 <= 250ms): {'PASS' if report['slo_p95_250ms'] else 'FAIL'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
