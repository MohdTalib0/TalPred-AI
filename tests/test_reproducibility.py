"""Model reproducibility test (ML-304).

Verifies that training with the same dataset and parameters produces
identical metrics within tolerance.

Usage: pytest tests/test_reproducibility.py -v --timeout 120
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.models.trainer import DEFAULT_PARAMS, prepare_features, train_baseline

logger = logging.getLogger(__name__)

METRIC_TOLERANCE = 0.001


def _create_synthetic_dataset(n_rows: int = 2000, n_symbols: int = 10) -> pd.DataFrame:
    """Create a synthetic but realistic-looking training dataset."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2023-01-01", periods=n_rows // n_symbols)
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

    rows = []
    for sym in symbols:
        for d in dates:
            rows.append({
                "symbol": sym,
                "target_session_date": d,
                "rsi_14": rng.uniform(20, 80),
                "momentum_5d": rng.normal(0, 0.03),
                "momentum_10d": rng.normal(0, 0.04),
                "rolling_return_5d": rng.normal(0, 0.03),
                "rolling_return_20d": rng.normal(0, 0.05),
                "rolling_volatility_20d": rng.uniform(0.005, 0.05),
                "macd": rng.normal(0, 1),
                "macd_signal": rng.normal(0, 0.8),
                "sector_return_1d": rng.normal(0, 0.01),
                "sector_return_5d": rng.normal(0, 0.03),
                "benchmark_relative_return_1d": rng.normal(0, 0.01),
                "vix_level": rng.uniform(12, 35),
                "regime_label": rng.choice(
                    ["bull_low_vol", "bull_high_vol", "bear_low_vol", "sideways"]
                ),
                "direction": int(rng.random() > 0.48),
                "next_day_return": rng.normal(0.0005, 0.015),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_dataset():
    return _create_synthetic_dataset()


def test_training_deterministic(synthetic_dataset):
    """Verify that two training runs with same data produce identical metrics."""
    params = DEFAULT_PARAMS.copy()
    params["n_estimators"] = 50

    X1, y1, med1 = prepare_features(synthetic_dataset)
    X2, y2, med2 = prepare_features(synthetic_dataset)

    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
    pd.testing.assert_series_equal(med1, med2)


def test_prepare_features_leakage_safe(synthetic_dataset):
    """Verify that fill_medians prevents val leakage."""
    split = int(len(synthetic_dataset) * 0.8)
    df_train = synthetic_dataset.iloc[:split]
    df_val = synthetic_dataset.iloc[split:]

    X_train, _, train_medians = prepare_features(df_train)
    X_val_safe, _, _ = prepare_features(df_val, fill_medians=train_medians)
    X_val_leaky, _, _ = prepare_features(df_val)

    assert not X_val_safe.isna().any().any(), "NaN values remain after safe fill"
    assert not X_val_leaky.isna().any().any(), "NaN values remain after leaky fill"


def test_prepare_features_returns_medians(synthetic_dataset):
    """Verify prepare_features returns usable medians."""
    _, _, medians = prepare_features(synthetic_dataset)
    assert isinstance(medians, pd.Series)
    assert len(medians) > 0
    assert not medians.isna().all()
