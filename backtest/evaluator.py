"""Out-of-sample model evaluation and strategy simulation hooks."""

from __future__ import annotations

from typing import Sequence

from backtest.metrics import BacktestMetrics, calculate_metrics


def run_backtest(y_true: Sequence[float], y_pred: Sequence[float], strategy_returns: Sequence[float]) -> BacktestMetrics:
    """Run backtest evaluation over out-of-sample predictions."""
    return calculate_metrics(y_true=y_true, y_pred=y_pred, strategy_returns=strategy_returns)
