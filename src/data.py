"""Data acquisition and caching for market data."""

import hashlib
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    key = hashlib.md5(f"{ticker}_{start}_{end}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.parquet"


def fetch_ohlcv(
    ticker: str,
    start_date: str = "2023-01-01",
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data via yfinance with optional local parquet cache."""
    if end_date is None:
        end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    cache_file = _cache_path(ticker, start_date, end_date)

    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        repair=True,
    )

    if data.empty:
        raise ValueError(f"No data found for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_file)

    return data
