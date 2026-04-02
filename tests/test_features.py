"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, get_feature_matrix, FEATURE_COLS


def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_build_features_adds_columns():
    raw = _make_ohlcv()
    df = build_features(raw)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_feature_matrix_no_nan():
    raw = _make_ohlcv()
    df = build_features(raw)
    fm = get_feature_matrix(df)
    assert fm.isna().sum().sum() == 0
    assert not np.isinf(fm.values).any()


def test_feature_matrix_length():
    raw = _make_ohlcv(200)
    df = build_features(raw)
    fm = get_feature_matrix(df)
    # Should lose ~50 rows to warmup (SMA50 needs 50 bars)
    assert 100 < len(fm) < 200
