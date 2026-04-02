"""Regime detection via Hidden Markov Models.

HMM advantages over K-Means and GMM:
- Captures temporal persistence (regimes tend to last days/weeks)
- Transition matrix reveals regime change dynamics
- Naturally models sequential market data
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class HMMResult:
    ticker: str
    n_states: int
    log_likelihood_train: float
    log_likelihood_test: float
    bic_train: float
    silhouette_train: float
    silhouette_test: float
    transition_matrix: np.ndarray
    stationary_distribution: np.ndarray
    cluster_profile: pd.DataFrame
    cluster_sizes: pd.Series
    model: GaussianHMM
    scaler: StandardScaler
    bic_scores: dict[int, float] = field(default_factory=dict)


def _bic_hmm(model: GaussianHMM, X: np.ndarray) -> float:
    """Compute BIC for an HMM model.

    n_params = n_states^2 (transition) + n_states*n_features (means)
             + n_states*n_features (diagonal covariance) + n_states-1 (start probs)
    """
    n_samples, n_features = X.shape
    n_states = model.n_components
    # For "diag" covariance type
    n_params = (
        n_states * n_states       # transition matrix
        + n_states * n_features   # means
        + n_states * n_features   # diagonal variances
        + (n_states - 1)          # start probabilities
    )
    ll = model.score(X) * n_samples  # score returns per-sample avg log-likelihood
    return -2 * ll + n_params * np.log(n_samples)


def select_n_states(
    X_train: np.ndarray,
    n_min: int = 2,
    n_max: int = 6,
) -> tuple[int, dict[int, float]]:
    """Select optimal number of hidden states using BIC."""
    max_n = min(n_max, len(X_train) // 20)
    if max_n < n_min:
        raise ValueError("Not enough samples for HMM evaluation")

    bic_scores: dict[int, float] = {}
    for n in range(n_min, max_n + 1):
        best_bic = np.inf
        # Multiple random starts to avoid local optima
        for init_seed in range(5):
            hmm = GaussianHMM(
                n_components=n,
                covariance_type="diag",
                n_iter=200,
                random_state=init_seed,
                tol=1e-4,
            )
            try:
                hmm.fit(X_train)
                bic = _bic_hmm(hmm, X_train)
                if bic < best_bic:
                    best_bic = bic
            except Exception:
                continue
        bic_scores[n] = best_bic

    if all(np.isinf(v) for v in bic_scores.values()):
        raise ValueError("HMM fitting failed for all n_states")

    n_opt = min(bic_scores, key=bic_scores.get)
    return n_opt, bic_scores


def fit_hmm(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    test_size: float = 0.2,
    n_min: int = 2,
    n_max: int = 6,
) -> tuple[GaussianHMM, StandardScaler, int, dict[int, float], float, float, float, float]:
    """Train HMM on train split, evaluate on test.

    Returns (model, scaler, n_opt, bic_scores, bic_train, ll_train, sil_train, sil_test).
    """
    if len(df_model) < 60:
        raise ValueError(f"Too few observations: {len(df_model)}")

    split_idx = int(len(df_model) * (1 - test_size))
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    n_opt, bic_scores = select_n_states(X_train, n_min, n_max)

    # Final model with more iterations and multiple starts
    best_model = None
    best_ll = -np.inf
    for init_seed in range(10):
        hmm = GaussianHMM(
            n_components=n_opt,
            covariance_type="diag",
            n_iter=500,
            random_state=init_seed,
            tol=1e-5,
        )
        try:
            hmm.fit(X_train)
            ll = hmm.score(X_train)
            if ll > best_ll:
                best_ll = ll
                best_model = hmm
        except Exception:
            continue

    if best_model is None:
        raise ValueError(f"HMM fitting failed for n_states={n_opt}")

    model = best_model
    bic_train = _bic_hmm(model, X_train)

    # Silhouette on train
    train_labels = model.predict(X_train)
    if len(np.unique(train_labels)) >= 2:
        sil_train = silhouette_score(X_train, train_labels)
    else:
        sil_train = np.nan

    # Silhouette on test (uses Viterbi decode, respects temporal structure)
    test_labels = model.predict(X_test)
    if len(np.unique(test_labels)) >= 2:
        sil_test = silhouette_score(X_test, test_labels)
    else:
        sil_test = np.nan

    ll_train = model.score(X_train) * len(X_train)

    return model, scaler, n_opt, bic_scores, bic_train, ll_train, sil_train, sil_test


def label_hmm(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    model: GaussianHMM,
    scaler: StandardScaler,
) -> tuple[pd.Series, np.ndarray]:
    """Assign labels via Viterbi decoding and posterior probabilities."""
    X = scaler.transform(df_model[feature_cols])
    labels = pd.Series(model.predict(X), index=df_model.index, name="Cluster")
    probs = model.predict_proba(X)
    return labels, probs


def get_transition_matrix(model: GaussianHMM) -> pd.DataFrame:
    """Return transition matrix as a labeled DataFrame."""
    n = model.n_components
    names = [f"Regime_{i}" for i in range(n)]
    return pd.DataFrame(model.transmat_, index=names, columns=names).round(4)


def get_stationary_distribution(model: GaussianHMM) -> np.ndarray:
    """Compute stationary distribution from transition matrix."""
    transmat = model.transmat_
    eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
    # Find eigenvector corresponding to eigenvalue ≈ 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()
    return stationary
