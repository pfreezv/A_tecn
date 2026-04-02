"""Generate realistic synthetic OHLCV data with embedded regimes for testing."""

import numpy as np
import pandas as pd


def generate_regime_data(
    n_days: int = 600,
    start_date: str = "2022-01-03",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate OHLCV data with 3 embedded market regimes.

    Returns (ohlcv_df, true_regimes) where true_regimes is a Series
    with labels 0=bull, 1=sideways, 2=bear for validation.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_days)

    # Regime sequence proportions: bull → sideways → bear → bull → bear → bull
    regime_fractions = [
        (0.00, 0.20, 0),   # bull
        (0.20, 0.37, 1),   # sideways
        (0.37, 0.53, 2),   # bear
        (0.53, 0.73, 0),   # bull
        (0.73, 0.85, 2),   # bear
        (0.85, 1.00, 0),   # bull
    ]
    regime_blocks = [
        (int(f0 * n_days), int(f1 * n_days), r)
        for f0, f1, r in regime_fractions
    ]

    # Regime parameters: (daily_drift, daily_vol, volume_base, volume_vol)
    params = {
        0: (0.0012, 0.012, 8_000_000, 2_000_000),   # bull: positive drift, moderate vol
        1: (0.0001, 0.008, 5_000_000, 1_000_000),    # sideways: near-zero drift, low vol
        2: (-0.0015, 0.022, 12_000_000, 4_000_000),  # bear: negative drift, high vol
    }

    returns = np.zeros(n_days)
    volumes = np.zeros(n_days)
    true_labels = np.zeros(n_days, dtype=int)

    for start, end, regime in regime_blocks:
        drift, vol, vol_base, vol_std = params[regime]
        length = end - start
        returns[start:end] = rng.normal(drift, vol, length)
        volumes[start:end] = rng.normal(vol_base, vol_std, length).clip(500_000)
        true_labels[start:end] = regime

    # Build price series
    close = 150 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    intraday_range = np.abs(returns) + rng.uniform(0.002, 0.008, n_days)
    high = close * (1 + intraday_range / 2)
    low = close * (1 - intraday_range / 2)
    open_ = close * (1 + rng.normal(0, 0.003, n_days))

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volumes,
        },
        index=dates,
    )

    true_regimes = pd.Series(true_labels, index=dates, name="TrueRegime")
    return df, true_regimes
