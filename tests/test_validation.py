from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.pipelines.validation import check_nulls, check_outliers, compute_freshness_check


def test_check_nulls_finds_missing():
    df = pd.DataFrame({
        "a": [1, 2, None],
        "b": ["x", None, "z"],
    })
    bad = check_nulls(df, ["a", "b"])
    assert len(bad) == 2


def test_check_nulls_no_issues():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    })
    bad = check_nulls(df, ["a", "b"])
    assert len(bad) == 0


def test_check_outliers_detects_extreme():
    s = pd.Series([10, 11, 12, 10, 11, 100])
    outliers = check_outliers(s, z_threshold=3.0)
    assert outliers.iloc[-1] is True or outliers.iloc[-1] == True


def test_freshness_passes():
    recent = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    result = compute_freshness_check(recent, sla_hours=30)
    assert result["fresh"] is True


def test_freshness_fails():
    old = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=48)
    result = compute_freshness_check(old, sla_hours=30)
    assert result["fresh"] is False


def test_freshness_no_data():
    result = compute_freshness_check(None)
    assert result["fresh"] is False
