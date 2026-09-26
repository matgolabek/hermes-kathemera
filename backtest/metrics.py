"""Financial performance metrics for backtests."""

from __future__ import annotations

from dataclasses import dataclass

import ccxt
import numpy as np

SECONDS_PER_YEAR = 365 * 24 * 3600  # crypto markets trade every day


@dataclass(frozen=True)
class BacktestMetrics:
    """Summary of a strategy's per-candle returns.

    Attributes:
        total_return: Compounded return over the whole period.
        annual_return: Compounded return scaled to one year.
        annual_volatility: Standard deviation of returns scaled to one year.
        sharpe: Annualized mean return divided by volatility (risk-free rate 0).
        max_drawdown: Largest fall from a previous equity peak (negative).
        exposure: Share of candles with an open position.
        trades: Number of position changes, including the final close.
        hit_rate: Share of in-market candles with a positive return.
        costs: Sum of fees and slippage, as a fraction of equity per candle.
    """

    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    exposure: float
    trades: int
    hit_rate: float
    costs: float


def periods_per_year(timeframe: str) -> float:
    """Number of candles of ``timeframe`` (e.g. ``"1h"``) in a year of 24/7 trading."""
    return SECONDS_PER_YEAR / ccxt.Exchange.parse_timeframe(timeframe)


def calculate_metrics(
    net_returns: np.ndarray, positions: np.ndarray, costs: np.ndarray, trades: int, periods: float
) -> BacktestMetrics:
    """Compute performance metrics from per-candle simple returns after costs."""
    returns = np.asarray(net_returns, dtype=np.float64)
    if returns.size == 0:
        raise ValueError("Cannot compute metrics for an empty backtest")

    equity = np.cumprod(1.0 + returns)
    drawdown = equity / np.maximum.accumulate(np.maximum(equity, 1.0)) - 1.0
    years = returns.size / periods
    std = returns.std(ddof=1) if returns.size > 1 else 0.0
    in_market = np.asarray(positions) != 0

    return BacktestMetrics(
        total_return=float(equity[-1] - 1.0),
        annual_return=float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0,
        annual_volatility=float(std * np.sqrt(periods)),
        sharpe=float(returns.mean() / std * np.sqrt(periods)) if std > 0 else 0.0,
        max_drawdown=float(drawdown.min()),
        exposure=float(in_market.mean()),
        trades=int(trades),
        hit_rate=float((returns[in_market] > 0).mean()) if in_market.any() else 0.0,
        costs=float(np.sum(costs)),
    )
