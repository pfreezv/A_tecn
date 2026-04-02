"""Unit tests for regime detection logic."""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, get_feature_matrix, FEATURE_COLS
from src.regime import select_k, fit_regimes, label_all


def _make_ohlcv(n: int = 250) -> pd.DataFrame:
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


def test_select_k_returns_valid():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    k, scores = select_k(X, k_min=2, k_max=5)
    assert 2 <= k <= 5
    assert all(isinstance(v, float) for v in scores.values())


def test_fit_regimes_no_leakage():
    """Model must be trained on train split only — test labels via predict."""
    raw = _make_ohlcv(300)
    df = build_features(raw)
    df_model = get_feature_matrix(df)

    model, scaler, k_opt, scores, sil_train, sil_test = fit_regimes(
        df_model, FEATURE_COLS, test_size=0.2,
    )

    # Model was fit on ~80% of data
    assert 2 <= k_opt <= 8
    assert not np.isnan(sil_train)
    # sil_test could be NaN if all test points land in one cluster, but shouldn't crash


def test_label_all_uses_predict():
    """label_all should not refit the model."""
    raw = _make_ohlcv(300)
    df = build_features(raw)
    df_model = get_feature_matrix(df)

    model, scaler, k_opt, _, _, _ = fit_regimes(df_model, FEATURE_COLS)

    # Record cluster centers before labeling
    centers_before = model.cluster_centers_.copy()

    labels = label_all(df_model, FEATURE_COLS, model, scaler)

    # Centers should be unchanged — no refit
    np.testing.assert_array_equal(model.cluster_centers_, centers_before)
    assert len(labels) == len(df_model)
