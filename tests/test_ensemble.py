"""Tests for the ensemble regime detection system."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

from src.synthetic import generate_regime_data
from src.ensemble import fit_ensemble, _classify_cluster_semantics


def _get_data():
    raw, true_regimes = generate_regime_data(n_days=300, seed=99)
    return raw, true_regimes


def test_ensemble_runs_end_to_end():
    raw, _ = _get_data()
    result = fit_ensemble("TEST", raw, test_size=0.2, k_min=2, k_max=3, show_progress=False)

    assert result.ticker == "TEST"
    assert len(result.consensus) > 100
    assert set(result.consensus.unique()).issubset({"bull", "bear", "sideways"})
    assert all(result.confidence.between(0.33, 1.0))


def test_signal_strength_values():
    raw, _ = _get_data()
    result = fit_ensemble("TEST", raw, test_size=0.2, k_min=2, k_max=3, show_progress=False)

    valid_strengths = {"strong", "moderate", "weak"}
    assert set(result.signal_strength.unique()).issubset(valid_strengths)


def test_models_produce_different_labels():
    """Verify models are actually independent, not all returning the same thing."""
    raw, _ = _get_data()
    result = fit_ensemble("TEST", raw, test_size=0.2, k_min=2, k_max=4, show_progress=False)

    # At least one pair of models should disagree on some days
    km_vs_gmm = (result.kmeans_semantic != result.gmm_semantic).sum()
    km_vs_hmm = (result.kmeans_semantic != result.hmm_semantic).sum()
    total_disagree = km_vs_gmm + km_vs_hmm
    assert total_disagree > 0, "All models agree on everything — suspicious"


def test_classify_cluster_semantics():
    profile = pd.DataFrame({
        "ret_1d": [0.002, -0.003, 0.0001],
        "vol_20": [0.01, 0.025, 0.006],
        "ret_5d": [0, 0, 0],
        "RSI_14": [55, 35, 50],
        "sma_spread_pct": [0, 0, 0],
        "dist_sma50_pct": [0, 0, 0],
        "atr_pct": [0, 0, 0],
        "volume_ratio_20": [1, 1, 1],
    }, index=[0, 1, 2])

    mapping = _classify_cluster_semantics(profile)
    assert mapping[0] == "bull"
    assert mapping[1] == "bear"
    assert mapping[2] == "sideways"


def test_hmm_transition_matrix_valid():
    raw, _ = _get_data()
    result = fit_ensemble("TEST", raw, test_size=0.2, k_min=2, k_max=3, show_progress=False)

    tm = result.transition_matrix.values
    # Rows should sum to ~1
    row_sums = tm.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=0.01)
    # All values non-negative
    assert (tm >= 0).all()
