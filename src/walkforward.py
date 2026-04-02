"""Walk-forward regime detection: re-trains models periodically.

Instead of a single train/test split, this slides a window forward
and retrains every `retrain_every` days, simulating real-world deployment.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import silhouette_score


def _fit_kmeans(X_train: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_train)
    return model


def _fit_gmm(X_train: np.ndarray, k: int) -> GaussianMixture:
    model = GaussianMixture(
        n_components=k, covariance_type="full",
        random_state=42, n_init=5, max_iter=300,
    )
    model.fit(X_train)
    return model


def _fit_hmm(X_train: np.ndarray, n: int) -> GaussianHMM:
    best_model, best_ll = None, -np.inf
    for seed in range(5):
        hmm = GaussianHMM(
            n_components=n, covariance_type="diag",
            n_iter=300, random_state=seed, tol=1e-4,
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
        raise ValueError(f"HMM fitting failed for n_states={n}")
    return best_model


def _classify_regime(profile_row: pd.Series, median_vol: float) -> str:
    ret = profile_row["ret_1d"]
    vol = profile_row["vol_20"]
    if ret > 0.0005:
        return "bull"
    elif ret < -0.0005 and vol > median_vol:
        return "bear"
    return "sideways"


def walk_forward(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    min_train: int = 120,
    retrain_every: int = 20,
    k_kmeans: int = 3,
    k_gmm: int = 3,
    n_hmm: int = 3,
) -> pd.DataFrame:
    """Walk-forward regime labeling.

    For each evaluation window:
    1. Train on all data up to current point
    2. Predict the next `retrain_every` days
    3. Slide forward and repeat

    Returns DataFrame with columns:
    - KM_regime, GMM_regime, HMM_regime: semantic labels per model
    - Consensus: majority vote
    - Confidence: agreement fraction
    """
    n = len(df_model)
    if n < min_train + retrain_every:
        raise ValueError(f"Need at least {min_train + retrain_every} samples, got {n}")

    results = []

    for train_end in range(min_train, n, retrain_every):
        eval_end = min(train_end + retrain_every, n)
        train_data = df_model.iloc[:train_end]
        eval_data = df_model.iloc[train_end:eval_end]

        if len(eval_data) == 0:
            break

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[feature_cols])
        X_eval = scaler.transform(eval_data[feature_cols])

        # Fit all three models on training data
        try:
            km = _fit_kmeans(X_train, k_kmeans)
            km_train_labels = km.predict(X_train)
            km_eval_labels = km.predict(X_eval)
        except Exception:
            km_eval_labels = np.full(len(eval_data), -1)
            km_train_labels = np.full(len(train_data), -1)

        try:
            gmm = _fit_gmm(X_train, k_gmm)
            gmm_train_labels = gmm.predict(X_train)
            gmm_eval_labels = gmm.predict(X_eval)
        except Exception:
            gmm_eval_labels = np.full(len(eval_data), -1)
            gmm_train_labels = np.full(len(train_data), -1)

        try:
            hmm = _fit_hmm(X_train, n_hmm)
            hmm_train_labels = hmm.predict(X_train)
            hmm_eval_labels = hmm.predict(X_eval)
        except Exception:
            hmm_eval_labels = np.full(len(eval_data), -1)
            hmm_train_labels = np.full(len(train_data), -1)

        # Build semantic maps from training profiles
        for model_name, train_labels, eval_labels in [
            ("KM", km_train_labels, km_eval_labels),
            ("GMM", gmm_train_labels, gmm_eval_labels),
            ("HMM", hmm_train_labels, hmm_eval_labels),
        ]:
            # Profile from training data
            train_df_tmp = train_data[feature_cols].copy()
            train_df_tmp["_label"] = train_labels
            profile = train_df_tmp.groupby("_label")[feature_cols].mean()
            median_vol = profile["vol_20"].median() if "vol_20" in profile.columns else 0.01

            sem_map = {}
            for cid, row in profile.iterrows():
                if cid == -1:
                    sem_map[cid] = "unknown"
                else:
                    sem_map[cid] = _classify_regime(row, median_vol)

            # Apply to eval labels
            for i, idx in enumerate(eval_data.index):
                label = eval_labels[i]
                sem = sem_map.get(label, "unknown")
                # Find or create entry
                found = False
                for r in results:
                    if r["date"] == idx:
                        r[f"{model_name}_regime"] = sem
                        found = True
                        break
                if not found:
                    results.append({
                        "date": idx,
                        f"{model_name}_regime": sem,
                    })

    # Build DataFrame
    wf = pd.DataFrame(results).set_index("date").sort_index()

    # Fill missing columns
    for col in ["KM_regime", "GMM_regime", "HMM_regime"]:
        if col not in wf.columns:
            wf[col] = "unknown"

    # Consensus
    def _vote(row):
        votes = [row["KM_regime"], row["GMM_regime"], row["HMM_regime"]]
        votes = [v for v in votes if v != "unknown"]
        if not votes:
            return "unknown"
        from collections import Counter
        return Counter(votes).most_common(1)[0][0]

    wf["Consensus"] = wf.apply(_vote, axis=1)
    wf["Confidence"] = wf.apply(
        lambda row: sum(1 for c in ["KM_regime", "GMM_regime", "HMM_regime"]
                        if row[c] == row["Consensus"]) / 3.0,
        axis=1,
    )
    wf["SignalStrength"] = wf["Confidence"].map(
        lambda c: "strong" if c == 1.0 else ("moderate" if c >= 0.67 else "weak")
    )

    return wf
