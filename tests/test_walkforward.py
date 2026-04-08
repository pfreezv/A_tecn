"""Tests for walk-forward regime labeling."""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pytest

from src.synthetic import generate_regime_data
from src.features import build_features, get_feature_matrix, FEATURE_COLS
from src.walkforward import walk_forward


def _prep(n_days: int = 400):
    raw, _ = generate_regime_data(n_days=n_days, seed=11)
    df_model = get_feature_matrix(build_features(raw))
    return df_model


def test_walkforward_produces_expected_columns():
    df = _prep()
    wf = walk_forward(df, FEATURE_COLS, min_train=120, retrain_every=20)

    for col in ["KM_regime", "GMM_regime", "HMM_regime",
                "Consensus", "Confidence", "SignalStrength"]:
        assert col in wf.columns

    assert not wf.empty
    assert set(wf["Consensus"].unique()).issubset(
        {"bull", "bear", "sideways", "unknown"}
    )


def test_walkforward_no_lookahead():
    """Labels should only cover dates at/after min_train — not earlier."""
    df = _prep()
    wf = walk_forward(df, FEATURE_COLS, min_train=120, retrain_every=20)
    assert wf.index.min() >= df.index[120]
    assert wf.index.max() <= df.index[-1]


def test_walkforward_confidence_range():
    df = _prep()
    wf = walk_forward(df, FEATURE_COLS, min_train=120, retrain_every=20)
    assert wf["Confidence"].between(0.0, 1.0).all()
    assert (wf["SignalStrength"].isin(["strong", "moderate", "weak"])).all()


def test_walkforward_index_unique_and_sorted():
    """Each eval date should appear exactly once and be sorted ascending."""
    df = _prep()
    wf = walk_forward(df, FEATURE_COLS, min_train=120, retrain_every=20)
    assert wf.index.is_unique
    assert wf.index.is_monotonic_increasing


def test_walkforward_errors_on_small_data():
    df = _prep(n_days=200)
    with pytest.raises(ValueError):
        walk_forward(df, FEATURE_COLS, min_train=300, retrain_every=50)
