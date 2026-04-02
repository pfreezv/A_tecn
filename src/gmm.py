"""Regime detection via Gaussian Mixture Models.

GMM advantages over K-Means:
- Elliptical clusters (captures correlated features)
- Soft assignments (probability per regime, not hard labels)
- BIC/AIC for model selection (more principled than silhouette)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class GMMResult:
    ticker: str
    k_optimal: int
    bic_train: float
    silhouette_train: float
    silhouette_test: float
    probabilities: np.ndarray  # (n_samples, k) soft assignments
    cluster_profile: pd.DataFrame
    cluster_sizes: pd.Series
    model: GaussianMixture
    scaler: StandardScaler
    bic_scores: dict[int, float] = field(default_factory=dict)


def select_k_gmm(
    X_train: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
) -> tuple[int, dict[int, float]]:
    """Select optimal k using BIC (lower is better) on training data."""
    max_k = min(k_max, len(X_train) // 10)  # need ~10 samples per component
    if max_k < k_min:
        raise ValueError("Not enough samples for GMM evaluation")

    bic_scores: dict[int, float] = {}
    for k in range(k_min, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            n_init=5,
            max_iter=300,
        )
        gmm.fit(X_train)
        bic_scores[k] = gmm.bic(X_train)

    k_opt = min(bic_scores, key=bic_scores.get)
    return k_opt, bic_scores


def fit_gmm(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    test_size: float = 0.2,
    k_min: int = 2,
    k_max: int = 8,
) -> tuple[GaussianMixture, StandardScaler, int, dict[int, float], float, float, float]:
    """Train GMM on train split, evaluate on test.

    Returns (model, scaler, k_opt, bic_scores, bic_train, sil_train, sil_test).
    """
    if len(df_model) < 60:
        raise ValueError(f"Too few observations: {len(df_model)}")

    split_idx = int(len(df_model) * (1 - test_size))
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    k_opt, bic_scores = select_k_gmm(X_train, k_min, k_max)

    model = GaussianMixture(
        n_components=k_opt,
        covariance_type="full",
        random_state=42,
        n_init=5,
        max_iter=300,
    )
    model.fit(X_train)

    bic_train = model.bic(X_train)

    # Silhouette on train
    train_labels = model.predict(X_train)
    if len(np.unique(train_labels)) >= 2:
        sil_train = silhouette_score(X_train, train_labels)
    else:
        sil_train = np.nan

    # Silhouette on test
    test_labels = model.predict(X_test)
    if len(np.unique(test_labels)) >= 2:
        sil_test = silhouette_score(X_test, test_labels)
    else:
        sil_test = np.nan

    return model, scaler, k_opt, bic_scores, bic_train, sil_train, sil_test


def label_gmm(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    model: GaussianMixture,
    scaler: StandardScaler,
) -> tuple[pd.Series, np.ndarray]:
    """Assign labels and probabilities (soft assignments)."""
    X = scaler.transform(df_model[feature_cols])
    labels = pd.Series(model.predict(X), index=df_model.index, name="Cluster")
    probs = model.predict_proba(X)
    return labels, probs
