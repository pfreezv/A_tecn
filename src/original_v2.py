"""Original function preserved for reference and comparison."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .features import _sma, _rsi, _atr
import yfinance as yf
from datetime import date, timedelta
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def analizar_accion_y_clusterizar_v2(
    ticker: str,
    start_date: str = "2023-01-01",
    end_date: str | None = None,
    mostrar_graficos: bool = True,
    k_min: int = 2,
    k_max: int = 8,
    test_size: float = 0.2,
):
    print(f"--- Procesando {ticker} ---")

    if end_date is None:
        end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        repair=True,
    )

    if data.empty:
        raise ValueError(f"No se encontraron datos para {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data.copy()

    df["SMA_10"] = _sma(df["Close"], 10)
    df["SMA_50"] = _sma(df["Close"], 50)
    df["RSI_14"] = _rsi(df["Close"], 14)
    df["ATR_14"] = _atr(df["High"], df["Low"], df["Close"], 14)

    df["ret_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ret_5d"] = np.log(df["Close"] / df["Close"].shift(5))
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["sma_spread_pct"] = (df["SMA_10"] / df["SMA_50"]) - 1
    df["dist_sma50_pct"] = (df["Close"] / df["SMA_50"]) - 1
    df["atr_pct"] = df["ATR_14"] / df["Close"]
    df["volume_ratio_20"] = df["Volume"] / df["Volume"].rolling(20).mean()

    feature_cols = [
        "ret_1d",
        "ret_5d",
        "vol_20",
        "RSI_14",
        "sma_spread_pct",
        "dist_sma50_pct",
        "atr_pct",
        "volume_ratio_20",
    ]

    df_model = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if len(df_model) < 60:
        raise ValueError(f"Muy pocas observaciones útiles: {len(df_model)}")

    split_idx = int(len(df_model) * (1 - test_size))
    train = df_model.iloc[:split_idx].copy()

    if len(train) < max(k_min + 5, 20):
        raise ValueError("Muy pocos datos en train para clusterizar con fiabilidad")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train)
    X_all = scaler.transform(df_model)

    max_k_allowed = min(k_max, len(train) - 1, train.drop_duplicates().shape[0] - 1)
    if max_k_allowed < k_min:
        raise ValueError("No hay suficientes datos distintos para evaluar clusters")

    k_candidates = list(range(k_min, max_k_allowed + 1))
    silhouette_scores = []

    for k in k_candidates:
        model_k = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels_k = model_k.fit_predict(X_train)

        if len(np.unique(labels_k)) < 2:
            silhouette_scores.append(np.nan)
            continue

        silhouette_scores.append(silhouette_score(X_train, labels_k))

    if np.all(np.isnan(silhouette_scores)):
        raise ValueError("No se pudo calcular silhouette para ningún k")

    k_optimal = k_candidates[int(np.nanargmax(silhouette_scores))]
    best_score = float(np.nanmax(silhouette_scores))
    print(f"K óptimo: {k_optimal} | Silhouette train: {best_score:.4f}")

    model_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=20)
    labels_all = model_final.fit_predict(X_all)

    df_model["Cluster"] = labels_all
    df_result = df.join(df_model[["Cluster"]], how="left")

    cluster_profile = df_model.groupby("Cluster")[feature_cols].mean().round(4)
    cluster_sizes = df_model["Cluster"].value_counts(normalize=True).sort_index().round(4)

    if mostrar_graficos:
        plt.figure(figsize=(10, 4))
        plt.plot(k_candidates, silhouette_scores, marker="o")
        plt.title(f"Silhouette vs número de clusters - {ticker}")
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Silhouette score")
        plt.grid(True, alpha=0.3)
        plt.show()

        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(X_all)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            pca_components[:, 0],
            pca_components[:, 1],
            c=df_model["Cluster"],
            s=30,
            alpha=0.7,
        )
        plt.title(f"Clusters visualizados con PCA - {ticker}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True, alpha=0.3)
        plt.show()

        plt.figure(figsize=(14, 6))
        plt.plot(
            df_result.index, df_result["Close"], color="lightgray", alpha=0.7, label="Close"
        )
        for cluster_id in sorted(df_model["Cluster"].unique()):
            subset = df_result[df_result["Cluster"] == cluster_id]
            plt.scatter(subset.index, subset["Close"], s=14, label=f"Cluster {cluster_id}")
        plt.title(f"Regímenes detectados - {ticker}")
        plt.xlabel("Fecha")
        plt.ylabel("Precio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    print("\nPerfil medio por cluster:")
    print(cluster_profile)

    return {
        "ticker": ticker,
        "df": df_result,
        "df_model": df_model,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "model": model_final,
        "k_optimal": k_optimal,
        "silhouette_scores": dict(zip(k_candidates, silhouette_scores)),
        "cluster_profile": cluster_profile,
        "cluster_sizes": cluster_sizes,
    }
