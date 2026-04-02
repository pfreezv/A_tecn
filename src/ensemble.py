"""Ensemble regime detection: combines K-Means, GMM, and HMM signals.

Consensus logic:
- Each model votes on the current regime type (bull/bear/sideways)
- Votes are weighted by model confidence
- Agreement level determines signal strength
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

from .features import build_features, get_feature_matrix, FEATURE_COLS
from .regime import fit_regimes, label_all, compute_forward_returns
from .gmm import fit_gmm, label_gmm
from .hmm import fit_hmm, label_hmm, get_transition_matrix, get_stationary_distribution


# ---------- Regime semantic mapping ----------

def _classify_cluster_semantics(profile: pd.DataFrame) -> dict[int, str]:
    """Map cluster IDs to semantic labels based on feature means.

    Uses ret_1d (drift) and vol_20 (volatility) as primary classifiers:
    - Bull:     positive drift, any volatility
    - Bear:     negative drift, high volatility
    - Sideways: near-zero drift, low volatility
    """
    mapping = {}
    for cluster_id, row in profile.iterrows():
        ret = row["ret_1d"]
        vol = row["vol_20"]
        median_vol = profile["vol_20"].median()

        if ret > 0.0005:
            mapping[cluster_id] = "bull"
        elif ret < -0.0005 and vol > median_vol:
            mapping[cluster_id] = "bear"
        else:
            mapping[cluster_id] = "sideways"
    return mapping


# ---------- Ensemble result ----------

@dataclass
class EnsembleResult:
    """Combined output from all three models."""

    ticker: str

    # Per-model results
    kmeans_labels: pd.Series
    gmm_labels: pd.Series
    hmm_labels: pd.Series

    # Semantic labels (bull/bear/sideways)
    kmeans_semantic: pd.Series
    gmm_semantic: pd.Series
    hmm_semantic: pd.Series

    # Consensus
    consensus: pd.Series         # majority vote semantic label
    confidence: pd.Series        # fraction of models agreeing (0.33, 0.67, 1.0)
    signal_strength: pd.Series   # "strong" / "moderate" / "weak"

    # Model quality metrics
    metrics: pd.DataFrame

    # Profiles
    kmeans_profile: pd.DataFrame
    gmm_profile: pd.DataFrame
    hmm_profile: pd.DataFrame

    # Transition dynamics (HMM only)
    transition_matrix: pd.DataFrame
    stationary_dist: np.ndarray

    # Forward returns by consensus regime
    forward_returns: pd.DataFrame

    # Full dataframe
    df: pd.DataFrame


def _apply_semantic(labels: pd.Series, mapping: dict[int, str]) -> pd.Series:
    return labels.map(mapping).rename("Semantic")


def _majority_vote(row: pd.Series) -> str:
    """Return the most common label among models."""
    counts = row.value_counts()
    return counts.index[0]


def fit_ensemble(
    ticker: str,
    raw: pd.DataFrame,
    test_size: float = 0.2,
    k_min: int = 2,
    k_max: int = 6,
    show_progress: bool = True,
) -> EnsembleResult:
    """Run all three models and combine their signals."""

    # Features
    df = build_features(raw)
    df_model = get_feature_matrix(df)

    if show_progress:
        print(f"[Ensemble] {ticker}: {len(df_model)} samples, {len(FEATURE_COLS)} features")

    # --- K-Means ---
    if show_progress:
        print("[1/3] Fitting K-Means...")
    km_model, km_scaler, km_k, km_scores, km_sil_train, km_sil_test = fit_regimes(
        df_model, FEATURE_COLS, test_size, k_min, k_max,
    )
    km_labels = label_all(df_model, FEATURE_COLS, km_model, km_scaler)
    df_model["KM_Cluster"] = km_labels
    km_profile = df_model.groupby("KM_Cluster")[FEATURE_COLS].mean()
    km_semantic_map = _classify_cluster_semantics(km_profile)

    if show_progress:
        print(f"    K={km_k}, sil_train={km_sil_train:.3f}, sil_test={km_sil_test:.3f}")

    # --- GMM ---
    if show_progress:
        print("[2/3] Fitting GMM...")
    gmm_model, gmm_scaler, gmm_k, gmm_bics, gmm_bic, gmm_sil_train, gmm_sil_test = fit_gmm(
        df_model[FEATURE_COLS], FEATURE_COLS, test_size, k_min, k_max,
    )
    gmm_labels, gmm_probs = label_gmm(df_model, FEATURE_COLS, gmm_model, gmm_scaler)
    df_model["GMM_Cluster"] = gmm_labels
    gmm_profile = df_model.groupby("GMM_Cluster")[FEATURE_COLS].mean()
    gmm_semantic_map = _classify_cluster_semantics(gmm_profile)

    if show_progress:
        print(f"    K={gmm_k}, BIC={gmm_bic:.0f}, sil_train={gmm_sil_train:.3f}, sil_test={gmm_sil_test:.3f}")

    # --- HMM ---
    if show_progress:
        print("[3/3] Fitting HMM...")
    hmm_model, hmm_scaler, hmm_n, hmm_bics, hmm_bic, hmm_ll, hmm_sil_train, hmm_sil_test = fit_hmm(
        df_model[FEATURE_COLS], FEATURE_COLS, test_size, n_min=k_min, n_max=min(k_max, 6),
    )
    hmm_labels, hmm_probs = label_hmm(df_model, FEATURE_COLS, hmm_model, hmm_scaler)
    df_model["HMM_Cluster"] = hmm_labels
    hmm_profile = df_model.groupby("HMM_Cluster")[FEATURE_COLS].mean()
    hmm_semantic_map = _classify_cluster_semantics(hmm_profile)

    trans_matrix = get_transition_matrix(hmm_model)
    stat_dist = get_stationary_distribution(hmm_model)

    if show_progress:
        print(f"    N={hmm_n}, BIC={hmm_bic:.0f}, sil_train={hmm_sil_train:.3f}, sil_test={hmm_sil_test:.3f}")

    # --- Consensus ---
    if show_progress:
        print("[*] Computing consensus...")

    km_sem = _apply_semantic(km_labels, km_semantic_map)
    gmm_sem = _apply_semantic(gmm_labels, gmm_semantic_map)
    hmm_sem = _apply_semantic(hmm_labels, hmm_semantic_map)

    votes = pd.DataFrame({
        "KMeans": km_sem,
        "GMM": gmm_sem,
        "HMM": hmm_sem,
    }, index=df_model.index)

    consensus = votes.apply(_majority_vote, axis=1).rename("Consensus")
    confidence = votes.apply(
        lambda row: row.value_counts().iloc[0] / 3.0, axis=1
    ).rename("Confidence")
    signal_strength = confidence.map(
        lambda c: "strong" if c == 1.0 else ("moderate" if c >= 0.67 else "weak")
    ).rename("SignalStrength")

    # Merge into full df
    df_result = df.copy()
    df_result = df_result.join(df_model[["KM_Cluster", "GMM_Cluster", "HMM_Cluster"]], how="left")
    df_result["Consensus"] = consensus
    df_result["Confidence"] = confidence
    df_result["SignalStrength"] = signal_strength
    df_result["Cluster"] = consensus.map({"bull": 0, "sideways": 1, "bear": 2})

    # Forward returns by consensus regime
    fwd = compute_forward_returns(df_result)

    # Metrics summary
    metrics = pd.DataFrame({
        "Model": ["K-Means", "GMM", "HMM"],
        "N_Clusters": [km_k, gmm_k, hmm_n],
        "Sil_Train": [km_sil_train, gmm_sil_train, hmm_sil_train],
        "Sil_Test": [km_sil_test, gmm_sil_test, hmm_sil_test],
    }).set_index("Model")

    if show_progress:
        print("\n--- Model Comparison ---")
        print(metrics.round(4).to_string())
        print(f"\nConsensus regime distribution:")
        print(consensus.value_counts(normalize=True).round(3).to_string())

    return EnsembleResult(
        ticker=ticker,
        kmeans_labels=km_labels,
        gmm_labels=gmm_labels,
        hmm_labels=hmm_labels,
        kmeans_semantic=km_sem,
        gmm_semantic=gmm_sem,
        hmm_semantic=hmm_sem,
        consensus=consensus,
        confidence=confidence,
        signal_strength=signal_strength,
        metrics=metrics,
        kmeans_profile=km_profile.round(4),
        gmm_profile=gmm_profile.round(4),
        hmm_profile=hmm_profile.round(4),
        transition_matrix=trans_matrix,
        stationary_dist=stat_dist,
        forward_returns=fwd,
        df=df_result,
    )
