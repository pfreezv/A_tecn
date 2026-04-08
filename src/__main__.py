"""CLI entry point for the regime detection pipeline.

Usage examples:

    # Real data (requires yfinance)
    python -m src AAPL --start 2020-01-01 --end 2024-01-01

    # Synthetic data (no network required)
    python -m src --synthetic --synthetic-days 700

    # Disable walk-forward, increase transaction cost
    python -m src AAPL --no-walkforward --cost-bps 10
"""

import argparse
import logging
import sys
import warnings

from .pipeline import run_full_analysis


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src",
        description="Run the full regime detection + backtest pipeline.",
    )
    p.add_argument("ticker", nargs="?", default=None,
                   help="Ticker symbol (omit when using --synthetic).")
    p.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    p.add_argument("--synthetic", action="store_true",
                   help="Use the built-in synthetic OHLCV generator instead of yfinance.")
    p.add_argument("--synthetic-days", type=int, default=700,
                   help="Number of business days to generate (default: 700).")
    p.add_argument("--synthetic-seed", type=int, default=42,
                   help="RNG seed for the synthetic generator.")
    p.add_argument("--k-min", type=int, default=2, help="Minimum number of clusters.")
    p.add_argument("--k-max", type=int, default=6, help="Maximum number of clusters.")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of data used for the out-of-sample split.")
    p.add_argument("--no-walkforward", action="store_true",
                   help="Skip the walk-forward retraining pass.")
    p.add_argument("--wf-min-train", type=int, default=150,
                   help="Minimum training window for walk-forward.")
    p.add_argument("--wf-retrain-every", type=int, default=20,
                   help="Walk-forward retraining cadence (days).")
    p.add_argument("--cost-bps", type=float, default=5.0,
                   help="Transaction cost in basis points per unit of turnover.")
    p.add_argument("--plots", action="store_true",
                   help="Render matplotlib plots at the end of the run.")
    p.add_argument("--quiet", action="store_true", help="Suppress step-by-step logs.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    # hmmlearn prints non-convergence notices via logging, not warnings.
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)

    if args.synthetic:
        from .synthetic import generate_regime_data
        raw, _ = generate_regime_data(n_days=args.synthetic_days, seed=args.synthetic_seed)
        ticker = args.ticker or "SYNTH"
    else:
        if not args.ticker:
            parser.error("ticker is required unless --synthetic is used")
        raw = None
        ticker = args.ticker

    result = run_full_analysis(
        ticker=ticker,
        raw=raw,
        start_date=args.start,
        end_date=args.end,
        test_size=args.test_size,
        k_min=args.k_min,
        k_max=args.k_max,
        use_walkforward=not args.no_walkforward,
        wf_min_train=args.wf_min_train,
        wf_retrain_every=args.wf_retrain_every,
        cost_bps=args.cost_bps,
        show_plots=args.plots,
        show_progress=not args.quiet,
    )

    print("\n" + "=" * 72)
    print(f"FINAL COMPARISON — {result.ticker}")
    print("=" * 72)
    print(result.comparison.to_string())

    dist = result.ensemble.consensus.value_counts(normalize=True).round(3)
    print("\nEnsemble regime distribution:")
    print(dist.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
