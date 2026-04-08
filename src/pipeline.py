"""Main pipeline orchestrating data → features → regime detection → output."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .data import fetch_ohlcv
from .features import build_features, get_feature_matrix, FEATURE_COLS
from .regime import (
    RegimeResult,
    fit_regimes,
    label_all,
    compute_forward_returns,
)
from .ensemble import EnsembleResult, fit_ensemble
from .walkforward import walk_forward
from .backtest import (
    BacktestResult,
    run_backtest,
    run_buyhold,
    compare_strategies,
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
        from .visualize import (
            plot_silhouette_curve,
            plot_pca_clusters,
            plot_regimes_on_price,
            plot_forward_returns_heatmap,
        )
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


# ---------- Unified end-to-end pipeline ----------


@dataclass
class FullAnalysisResult:
    """Bundle with every output from the full regime analysis pipeline."""

    ticker: str
    raw: pd.DataFrame
    ensemble: EnsembleResult
    walkforward: Optional[pd.DataFrame]
    backtests: dict[str, BacktestResult]
    comparison: pd.DataFrame


def run_full_analysis(
    ticker: str,
    raw: pd.DataFrame | None = None,
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    test_size: float = 0.2,
    k_min: int = 2,
    k_max: int = 6,
    use_walkforward: bool = True,
    wf_min_train: int = 150,
    wf_retrain_every: int = 20,
    cost_bps: float = 5.0,
    show_plots: bool = False,
    show_progress: bool = True,
    use_cache: bool = True,
) -> FullAnalysisResult:
    """End-to-end analysis: data → ensemble → walk-forward → backtest.

    Args:
        ticker: Label for the analysis (also used to download data if `raw` is None).
        raw: Pre-loaded OHLCV dataframe. If None, data is fetched via yfinance.
        start_date / end_date: Date range for yfinance download.
        test_size / k_min / k_max: Forwarded to the ensemble fit.
        use_walkforward: If True, run walk-forward labeling and add it as a
            separate strategy in the backtest comparison.
        wf_min_train / wf_retrain_every: Walk-forward window parameters.
        cost_bps: Transaction cost (basis points) applied to the regime strategies.
        show_plots: If True, render the main charts and the equity curve plot.
        show_progress: Print per-step status.
    """
    # 1. Data
    if raw is None:
        if show_progress:
            print(f"[data] Fetching {ticker} {start_date}→{end_date or 'today'}")
        raw = fetch_ohlcv(ticker, start_date, end_date, use_cache=use_cache)
    else:
        if show_progress:
            print(f"[data] Using provided dataframe ({len(raw)} rows)")

    # 2. Ensemble regime detection
    ensemble = fit_ensemble(
        ticker, raw, test_size=test_size, k_min=k_min, k_max=k_max,
        show_progress=show_progress,
    )

    # 3. Optional walk-forward labels
    wf_df: pd.DataFrame | None = None
    if use_walkforward:
        df_model = get_feature_matrix(build_features(raw))
        min_required = wf_min_train + wf_retrain_every
        if len(df_model) < min_required:
            if show_progress:
                print(
                    f"[wf] Not enough samples ({len(df_model)} < {min_required}), "
                    "skipping walk-forward."
                )
        else:
            if show_progress:
                print(
                    f"[wf] Walk-forward: min_train={wf_min_train}, "
                    f"retrain_every={wf_retrain_every}"
                )
            wf_df = walk_forward(
                df_model, FEATURE_COLS,
                min_train=wf_min_train,
                retrain_every=wf_retrain_every,
                k_kmeans=3, k_gmm=3, n_hmm=3,
            )

    # 4. Backtests
    if show_progress:
        print("[bt] Running backtests...")
    prices = raw["Close"]

    bt_ensemble = run_backtest(
        prices, ensemble.consensus,
        strategy_name="Ensemble",
        cost_bps=cost_bps,
    )
    bt_bh = run_buyhold(prices)
    backtests: dict[str, BacktestResult] = {
        "Ensemble": bt_ensemble,
        "Buy & Hold": bt_bh,
    }
    ordered_results = [bt_ensemble, bt_bh]

    if wf_df is not None:
        bt_wf = run_backtest(
            prices, wf_df["Consensus"],
            strategy_name="Walk-Forward",
            cost_bps=cost_bps,
        )
        backtests["Walk-Forward"] = bt_wf
        # Show the out-of-sample strategy next to the in-sample ensemble.
        ordered_results.insert(1, bt_wf)

    comparison = compare_strategies(ordered_results)

    if show_progress:
        print("\n--- Backtest comparison ---")
        print(comparison.to_string())

    # 5. Plots (lazy-imported)
    if show_plots:
        from .visualize import (
            plot_regimes_on_price,
            plot_forward_returns_heatmap,
            plot_equity_curves,
        )
        plot_regimes_on_price(ensemble.df, ticker)
        plot_forward_returns_heatmap(ensemble.forward_returns, ticker)
        plot_equity_curves(ordered_results, ticker)

    return FullAnalysisResult(
        ticker=ticker,
        raw=raw,
        ensemble=ensemble,
        walkforward=wf_df,
        backtests=backtests,
        comparison=comparison,
    )
