"""Feature engineering for regime detection."""

import numpy as np
import pandas as pd
import pandas_ta as ta


FEATURE_COLS = [
    "ret_1d",
    "ret_5d",
    "vol_20",
    "RSI_14",
    "sma_spread_pct",
    "dist_sma50_pct",
    "atr_pct",
    "volume_ratio_20",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features on OHLCV dataframe.

    Returns a copy with indicator columns and the feature matrix columns
    (FEATURE_COLS) appended. Rows with NaN/inf in features are dropped.
    """
    df = df.copy()

    # Technical indicators
    df["SMA_10"] = ta.sma(df["Close"], length=10)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Derived features
    df["ret_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ret_5d"] = np.log(df["Close"] / df["Close"].shift(5))
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["sma_spread_pct"] = (df["SMA_10"] / df["SMA_50"]) - 1
    df["dist_sma50_pct"] = (df["Close"] / df["SMA_50"]) - 1
    df["atr_pct"] = df["ATR_14"] / df["Close"]
    df["volume_ratio_20"] = df["Volume"] / df["Volume"].rolling(20).mean()

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract clean feature matrix (no NaN, no inf) from featured dataframe."""
    return (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
