"""Visualization utilities for regime analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_silhouette_curve(
    silhouette_scores: dict[int, float],
    ticker: str,
    k_optimal: int,
) -> None:
    ks = list(silhouette_scores.keys())
    vals = list(silhouette_scores.values())

    plt.figure(figsize=(8, 4))
    plt.plot(ks, vals, marker="o", linewidth=2)
    plt.axvline(k_optimal, color="red", linestyle="--", alpha=0.6, label=f"k={k_optimal}")
    plt.title(f"Silhouette vs K — {ticker}")
    plt.xlabel("K")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pca_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    ticker: str,
) -> None:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum()

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=20, alpha=0.6, cmap="tab10")
    plt.title(f"PCA Clusters — {ticker} ({var_explained:.0%} variance explained)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_regimes_on_price(df: pd.DataFrame, ticker: str) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["Close"], color="lightgray", alpha=0.7, label="Close")

    for cid in sorted(df["Cluster"].dropna().unique()):
        subset = df[df["Cluster"] == cid]
        plt.scatter(subset.index, subset["Close"], s=12, label=f"Regime {int(cid)}")

    plt.title(f"Detected Regimes — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forward_returns_heatmap(
    fwd_df: pd.DataFrame,
    ticker: str,
) -> None:
    """Heatmap of mean forward returns by cluster and horizon."""
    means = fwd_df.xs("mean", level=1, axis=1)

    plt.figure(figsize=(10, 4))
    im = plt.imshow(means.values, aspect="auto", cmap="RdYlGn")
    plt.xticks(range(means.shape[1]), means.columns, rotation=45)
    plt.yticks(range(means.shape[0]), [f"Regime {i}" for i in means.index])
    plt.colorbar(im, label="Mean log-return")
    plt.title(f"Forward Returns by Regime — {ticker}")
    plt.tight_layout()
    plt.show()


def plot_equity_curves(results, ticker: str) -> None:
    """Plot equity curves for a list of BacktestResult objects."""
    plt.figure(figsize=(12, 6))
    for r in results:
        plt.plot(r.equity_curve.index, r.equity_curve.values,
                 label=f"{r.strategy_name} (Sharpe {r.sharpe_ratio:.2f})", linewidth=1.8)
    plt.title(f"Equity Curves — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
