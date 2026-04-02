"""Regime detection via clustering with proper train/test separation."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeResult:
    """Container for regime detection outputs."""

    ticker: str
    k_optimal: int
    silhouette_train: float
    silhouette_test: float
    cluster_profile: pd.DataFrame
    cluster_sizes: pd.Series
    forward_returns_by_cluster: pd.DataFrame
    df: pd.DataFrame
    df_model: pd.DataFrame
    feature_cols: list[str]
    scaler: StandardScaler
    model: KMeans
    silhouette_scores: dict[int, float] = field(default_factory=dict)


def select_k(
    X_train: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
) -> tuple[int, dict[int, float]]:
    """Find optimal k via silhouette score on training data only."""
    max_k = min(k_max, len(X_train) - 1, len(np.unique(X_train, axis=0)) - 1)
    if max_k < k_min:
        raise ValueError("Not enough distinct samples to evaluate clusters")

    scores: dict[int, float] = {}
    for k in range(k_min, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_train)
        if len(np.unique(labels)) < 2:
            scores[k] = np.nan
            continue
        scores[k] = silhouette_score(X_train, labels)

    if all(np.isnan(v) for v in scores.values()):
        raise ValueError("Could not compute silhouette for any k")

    k_opt = max(scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else -1)
    return k_opt, scores


def fit_regimes(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    test_size: float = 0.2,
    k_min: int = 2,
    k_max: int = 8,
) -> tuple[KMeans, StandardScaler, int, dict[int, float], float, float]:
    """Train regime model on train split, evaluate on test split.

    Returns (model, scaler, k_optimal, silhouette_scores, sil_train, sil_test).
    The model is trained ONLY on train data; test labels come from predict().
    """
    if len(df_model) < 60:
        raise ValueError(f"Too few observations: {len(df_model)}")

    split_idx = int(len(df_model) * (1 - test_size))
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]

    if len(train) < max(k_min + 5, 20):
        raise ValueError("Too few training samples for reliable clustering")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    k_opt, scores = select_k(X_train, k_min, k_max)

    # Final model — trained ONLY on training data
    model = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    model.fit(X_train)

    sil_train = scores[k_opt]

    # Out-of-sample evaluation
    test_labels = model.predict(X_test)
    if len(np.unique(test_labels)) >= 2:
        sil_test = silhouette_score(X_test, test_labels)
    else:
        sil_test = np.nan

    return model, scaler, k_opt, scores, sil_train, sil_test


def label_all(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    model: KMeans,
    scaler: StandardScaler,
) -> pd.Series:
    """Assign regime labels to all rows using predict (no refit)."""
    X = scaler.transform(df_model[feature_cols])
    return pd.Series(model.predict(X), index=df_model.index, name="Cluster")


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute forward log-returns per cluster for actionable analysis."""
    if horizons is None:
        horizons = [1, 5, 10, 20]

    for h in horizons:
        df[f"fwd_ret_{h}d"] = np.log(df["Close"].shift(-h) / df["Close"])

    fwd_cols = [f"fwd_ret_{h}d" for h in horizons]
    stats = df.dropna(subset=fwd_cols).groupby("Cluster")[fwd_cols].agg(["mean", "std", "count"])
    return stats
