"""Tests for the backtesting engine."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    run_backtest,
    run_buyhold,
    compare_strategies,
    DEFAULT_POSITION_MAP,
)


def _uptrend_prices(n: int = 252, drift: float = 0.001) -> pd.Series:
    dates = pd.bdate_range("2022-01-03", periods=n)
    returns = np.full(n, drift)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.Series(close, index=dates, name="Close")


def test_buyhold_positive_on_uptrend():
    prices = _uptrend_prices()
    bt = run_buyhold(prices)
    assert bt.total_return > 0
    assert bt.sharpe_ratio >= 0
    assert bt.n_trades == 1
    assert bt.total_cost == 0.0


def test_regime_strategy_all_bear_is_flat():
    """A fully-bear series should leave the strategy in cash."""
    prices = _uptrend_prices()
    regimes = pd.Series("bear", index=prices.index)
    bt = run_backtest(prices, regimes, strategy_name="AllBear")
    assert abs(bt.total_return) < 1e-9
    assert bt.n_trades == 0
    assert bt.max_drawdown == 0.0


def test_regime_strategy_all_bull_matches_buyhold():
    """100% long should recover almost the full buy-and-hold return."""
    prices = _uptrend_prices()
    regimes = pd.Series("bull", index=prices.index)
    bt = run_backtest(prices, regimes, strategy_name="AllBull")
    bh = run_buyhold(prices)
    # The regime strategy skips the first day (shift(1) lag), so it loses
    # roughly one day of drift vs buy & hold.
    assert abs(bt.total_return - bh.total_return) < 0.01


def test_transaction_costs_reduce_return():
    prices = _uptrend_prices()
    # Flip regime every 5 days to produce real turnover.
    regimes = pd.Series("bull", index=prices.index)
    regimes.iloc[::5] = "bear"

    free = run_backtest(prices, regimes, cost_bps=0.0)
    taxed = run_backtest(prices, regimes, cost_bps=50.0)

    assert taxed.total_return < free.total_return
    assert taxed.total_cost > 0.0
    assert free.total_cost == 0.0


def test_sortino_and_calmar_populated():
    prices = _uptrend_prices()
    regimes = pd.Series("bull", index=prices.index)
    bt = run_backtest(prices, regimes)
    assert bt.sortino_ratio >= 0
    # Uptrend with tiny vol may have max_dd == 0 → calmar is 0 by convention.
    assert bt.calmar_ratio >= 0


def test_position_map_sideways_half():
    """Sideways regime should apply 0.5 position sizing by default."""
    prices = _uptrend_prices(drift=0.002)
    regimes = pd.Series("sideways", index=prices.index)
    bt = run_backtest(prices, regimes)
    bh = run_buyhold(prices)
    # Half exposure → roughly half the buy-and-hold return.
    assert 0.35 * bh.total_return < bt.total_return < 0.65 * bh.total_return


def test_compare_strategies_table_columns():
    prices = _uptrend_prices()
    regimes = pd.Series("bull", index=prices.index)
    df = compare_strategies([run_backtest(prices, regimes), run_buyhold(prices)])
    assert len(df) == 2
    for col in ["Sharpe", "Sortino", "Calmar", "Max DD", "# Trades", "Cost"]:
        assert col in df.columns


def test_default_position_map_values():
    assert DEFAULT_POSITION_MAP["bull"] == 1.0
    assert DEFAULT_POSITION_MAP["sideways"] == 0.5
    assert DEFAULT_POSITION_MAP["bear"] == 0.0
    assert DEFAULT_POSITION_MAP["unknown"] == 0.0
