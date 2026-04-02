"""Main pipeline orchestrating data → features → regime detection → output."""

from .data import fetch_ohlcv
from .features import build_features, get_feature_matrix, FEATURE_COLS
from .regime import (
    RegimeResult,
    fit_regimes,
    label_all,
    compute_forward_returns,
)
from .visualize import (
    plot_silhouette_curve,
    plot_pca_clusters,
    plot_regimes_on_price,
    plot_forward_returns_heatmap,
)


def analyze_regime(
    ticker: str,
    start_date: str = "2023-01-01",
    end_date: str | None = None,
    show_plots: bool = True,
    k_min: int = 2,
    k_max: int = 8,
    test_size: float = 0.2,
    use_cache: bool = True,
) -> RegimeResult:
    """Full regime detection pipeline with proper train/test separation.

    Key differences from v2:
    - Model is trained ONLY on train split, test is predict-only (no leakage).
    - Out-of-sample silhouette is computed and reported.
    - Forward returns per cluster give actionable insight.
    - Data is cached locally to avoid redundant API calls.
    """
    print(f"--- Processing {ticker} ---")

    # 1. Data
    raw = fetch_ohlcv(ticker, start_date, end_date, use_cache=use_cache)

    # 2. Features
    df = build_features(raw)
    df_model = get_feature_matrix(df)

    # 3. Regime model (train-only fit)
    model, scaler, k_opt, scores, sil_train, sil_test = fit_regimes(
        df_model, FEATURE_COLS, test_size, k_min, k_max,
    )

    print(f"Optimal K: {k_opt} | Silhouette train: {sil_train:.4f} | test: {sil_test:.4f}")

    # 4. Label all data (predict, not fit)
    df_model["Cluster"] = label_all(df_model, FEATURE_COLS, model, scaler)
    df_result = df.join(df_model[["Cluster"]], how="left")

    # 5. Forward returns — the actionable output
    fwd_stats = compute_forward_returns(df_result)

    # 6. Cluster profiles
    cluster_profile = df_model.groupby("Cluster")[FEATURE_COLS].mean().round(4)
    cluster_sizes = df_model["Cluster"].value_counts(normalize=True).sort_index().round(4)

    print("\nCluster profile (feature means):")
    print(cluster_profile)
    print("\nForward returns by cluster (mean | std | count):")
    print(fwd_stats.round(4))

    # 7. Visualization
    if show_plots:
        plot_silhouette_curve(scores, ticker, k_opt)
        X_all = scaler.transform(df_model[FEATURE_COLS])
        plot_pca_clusters(X_all, df_model["Cluster"].values, ticker)
        plot_regimes_on_price(df_result, ticker)
        plot_forward_returns_heatmap(fwd_stats, ticker)

    return RegimeResult(
        ticker=ticker,
        k_optimal=k_opt,
        silhouette_train=sil_train,
        silhouette_test=sil_test,
        cluster_profile=cluster_profile,
        cluster_sizes=cluster_sizes,
        forward_returns_by_cluster=fwd_stats,
        df=df_result,
        df_model=df_model,
        feature_cols=FEATURE_COLS,
        scaler=scaler,
        model=model,
        silhouette_scores=scores,
    )
