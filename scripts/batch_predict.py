"""Run batch predictions for all symbols.

Usage:
  python -m scripts.batch_predict                          # latest date
  python -m scripts.batch_predict --date 2026-03-10        # specific date
  python -m scripts.batch_predict --no-explanations        # skip SHAP (faster)
"""

import argparse
import logging
import time
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.pipelines.batch_predict import run_batch_predictions  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch prediction runner")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--no-explanations", action="store_true", help="Skip SHAP explanations")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else None

    db = SessionLocal()
    t0 = time.time()

    try:
        result = run_batch_predictions(
            db,
            target_date=target_date,
            compute_explanations=not args.no_explanations,
        )
        elapsed = time.time() - t0

        logger.info(f"Batch prediction complete in {elapsed:.1f}s")
        logger.info(f"  Predictions: {result.get('predictions', 0)}")
        logger.info(f"  Cached: {result.get('cached', 0)}")
        logger.info(f"  Errors: {result.get('errors', 0)}")
        logger.info(f"  Model: {result.get('model_version', 'N/A')}")
        logger.info(f"  Date: {result.get('target_date', 'N/A')}")

    except RuntimeError as e:
        logger.error(f"Batch prediction failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
