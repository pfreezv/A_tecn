"""Feature engineering for regime detection.

All indicators computed manually — no pandas_ta dependency required.

V3: Added asymmetric features that help distinguish bull from bear regimes
(upside/downside volatility, skewness, drawdown).
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
    # New asymmetric features
    "upside_vol_20",
    "downside_vol_20",
    "vol_asymmetry",
    "skew_20",
    "drawdown_pct",
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
    (FEATURE_COLS) appended.
    """
    df = df.copy()

    # Technical indicators
    df["SMA_10"] = _sma(df["Close"], 10)
    df["SMA_50"] = _sma(df["Close"], 50)
    df["RSI_14"] = _rsi(df["Close"], 14)
    df["ATR_14"] = _atr(df["High"], df["Low"], df["Close"], 14)

    # Base features
    df["ret_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ret_5d"] = np.log(df["Close"] / df["Close"].shift(5))
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["sma_spread_pct"] = (df["SMA_10"] / df["SMA_50"]) - 1
    df["dist_sma50_pct"] = (df["Close"] / df["SMA_50"]) - 1
    df["atr_pct"] = df["ATR_14"] / df["Close"]
    df["volume_ratio_20"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- Asymmetric features (key for separating bull vs bear) ---

    # Upside vs downside volatility (20-day rolling)
    pos_ret = df["ret_1d"].clip(lower=0)
    neg_ret = df["ret_1d"].clip(upper=0)
    df["upside_vol_20"] = pos_ret.rolling(20).std()
    df["downside_vol_20"] = neg_ret.rolling(20).std()

    # Volatility asymmetry: >0 means more upside vol, <0 means more downside
    df["vol_asymmetry"] = df["upside_vol_20"] - df["downside_vol_20"]

    # Rolling skewness (20-day) — positive in bull, negative in bear
    df["skew_20"] = df["ret_1d"].rolling(20).skew()

    # Drawdown from rolling max (captures bear regime depth)
    rolling_max = df["Close"].cummax()
    df["drawdown_pct"] = (df["Close"] / rolling_max) - 1

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract clean feature matrix (no NaN, no inf) from featured dataframe."""
    return (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
