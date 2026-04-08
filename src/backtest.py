"""Simple regime-based backtester.

Strategy logic:
- bull  → 100% long
- sideways → 50% long (reduced exposure)
- bear  → 0% (cash)

Compares regime strategy vs buy-and-hold.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    strategy_name: str
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    total_cost: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.Series


# Default position sizing per regime
DEFAULT_POSITION_MAP = {
    "bull": 1.0,
    "sideways": 0.5,
    "bear": 0.0,
    "unknown": 0.0,
}


def run_backtest(
    prices: pd.Series,
    regimes: pd.Series,
    position_map: dict[str, float] | None = None,
    strategy_name: str = "Regime Strategy",
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Backtest a regime-based strategy.

    Args:
        prices: Close price series (DatetimeIndex).
        regimes: Regime labels aligned with prices ("bull", "bear", "sideways").
        position_map: Dict mapping regime name → position size (0.0 to 1.0).
        strategy_name: Label for this backtest.
        cost_bps: Round-trip-equivalent transaction cost in basis points applied
            to the absolute change in position each day (e.g. 5 = 0.05%).
    """
    if position_map is None:
        position_map = DEFAULT_POSITION_MAP

    # Align
    common_idx = prices.index.intersection(regimes.index)
    prices = prices.loc[common_idx].copy()
    regimes = regimes.loc[common_idx].copy()

    # Daily log returns
    log_ret = np.log(prices / prices.shift(1)).fillna(0)

    # Position: use yesterday's regime to avoid look-ahead bias
    positions = regimes.shift(1).map(position_map).fillna(0)

    # Transaction costs: proportional to turnover (|Δposition|)
    turnover = positions.diff().abs().fillna(positions.abs().iloc[0] if len(positions) else 0)
    cost_rate = cost_bps / 10_000.0
    cost_series = turnover * cost_rate

    # Strategy returns (net of costs)
    strat_ret = log_ret * positions - cost_series

    # Equity curve
    equity = np.exp(strat_ret.cumsum())
    equity = equity / equity.iloc[0]  # start at 1.0

    # Metrics
    n_days = len(strat_ret)
    n_years = n_days / 252

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    ann_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    ann_vol = float(strat_ret.std() * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Sortino: only downside volatility
    downside = strat_ret[strat_ret < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

    # Max drawdown
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1
    max_dd = float(drawdowns.min())

    # Calmar: annualized return / |max drawdown|
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    # Win rate (days with positive return when in position)
    in_market = strat_ret[positions > 0]
    win_rate = float((in_market > 0).mean()) if len(in_market) > 0 else 0.0

    # Number of regime changes (≈ trades) — ignore the initial position
    n_trades = int((positions.diff().abs() > 0).sum())

    return BacktestResult(
        strategy_name=strategy_name,
        total_return=total_return,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        win_rate=win_rate,
        n_trades=n_trades,
        total_cost=float(cost_series.sum()),
        equity_curve=equity,
        daily_returns=strat_ret,
        positions=positions,
    )


def run_buyhold(prices: pd.Series) -> BacktestResult:
    """Buy-and-hold benchmark."""
    log_ret = np.log(prices / prices.shift(1)).fillna(0)
    equity = np.exp(log_ret.cumsum())
    equity = equity / equity.iloc[0]

    n_days = len(log_ret)
    n_years = n_days / 252

    total_return = float(equity.iloc[-1] - 1)
    ann_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    ann_vol = float(log_ret.std() * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    downside = log_ret[log_ret < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1).min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    return BacktestResult(
        strategy_name="Buy & Hold",
        total_return=total_return,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        win_rate=float((log_ret > 0).mean()),
        n_trades=1,
        total_cost=0.0,
        equity_curve=equity,
        daily_returns=log_ret,
        positions=pd.Series(1.0, index=prices.index),
    )


def compare_strategies(results: list[BacktestResult]) -> pd.DataFrame:
    """Side-by-side comparison table."""
    rows = []
    for r in results:
        rows.append({
            "Strategy": r.strategy_name,
            "Total Return": f"{r.total_return:+.1%}",
            "Ann. Return": f"{r.annualized_return:+.1%}",
            "Ann. Vol": f"{r.annualized_vol:.1%}",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "Sortino": f"{r.sortino_ratio:.2f}",
            "Calmar": f"{r.calmar_ratio:.2f}",
            "Max DD": f"{r.max_drawdown:.1%}",
            "Win Rate": f"{r.win_rate:.1%}",
            "# Trades": r.n_trades,
            "Cost": f"{r.total_cost:.2%}",
        })
    return pd.DataFrame(rows).set_index("Strategy")
