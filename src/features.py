"""Feature engineering for regime detection.

All indicators computed manually — no pandas_ta dependency required.
"""

import numpy as np
import pandas as pd


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


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features on OHLCV dataframe.

    Returns a copy with indicator columns and the feature matrix columns
    (FEATURE_COLS) appended. Rows with NaN/inf in features are dropped.
    """
    df = df.copy()

    # Technical indicators (manual — no external TA library needed)
    df["SMA_10"] = _sma(df["Close"], 10)
    df["SMA_50"] = _sma(df["Close"], 50)
    df["RSI_14"] = _rsi(df["Close"], 14)
    df["ATR_14"] = _atr(df["High"], df["Low"], df["Close"], 14)

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
