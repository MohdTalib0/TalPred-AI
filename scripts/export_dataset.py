"""Export training dataset for DVC versioning (ML-302).

Exports features + labels to a versioned Parquet file,
then tracks it with DVC for reproducible lineage.

Usage:
  python -m scripts.export_dataset
  python -m scripts.export_dataset --output data/datasets/train_v1.parquet
"""

import argparse
import logging
import os
import time
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from src.db import SessionLocal  # noqa: E402
from src.features.leakage import build_training_dataset, validate_no_leakage  # noqa: E402
from src.models.schema import Symbol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export training dataset")
    parser.add_argument(
        "--output", type=str,
        default="data/datasets/training_latest.parquet",
    )
    parser.add_argument("--years", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    db = SessionLocal()
    t0 = time.time()

    try:
        symbols = [
            row.symbol for row in
            db.query(Symbol).filter(Symbol.is_active.is_(True)).all()
        ]
        end_date = date.today()
        start_date = end_date - timedelta(days=args.years * 365)

        logger.info(f"Building dataset: {len(symbols)} symbols, {start_date} -> {end_date}")
        df = build_training_dataset(db, symbols, start_date, end_date)

        if df.empty:
            logger.error("No data. Generate features first.")
            return

        leakage = validate_no_leakage(df)
        logger.info(f"Leakage check: {'PASSED' if leakage['passed'] else 'FAILED'}")

        df.to_parquet(args.output, index=False)
        elapsed = time.time() - t0

        logger.info(f"Exported {len(df)} rows to {args.output} in {elapsed:.1f}s")
        logger.info(f"  Symbols: {df['symbol'].nunique()}")
        logger.info(f"  Date range: {df['target_session_date'].min()} to {df['target_session_date'].max()}")
        logger.info(f"  File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
        logger.info(f"\nTo version with DVC:")
        logger.info(f"  dvc add {args.output}")
        logger.info(f"  git add {args.output}.dvc .gitignore")
        logger.info(f"  git commit -m 'Add training dataset'")
        logger.info(f"  dvc push")

    finally:
        db.close()


if __name__ == "__main__":
    main()
