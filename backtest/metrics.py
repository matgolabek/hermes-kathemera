"""Financial and prediction metrics for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BacktestMetrics:
    """Container for key backtesting summary metrics."""

    mse: float
    directional_accuracy: float
    cumulative_return: float
    max_drawdown: float


def calculate_metrics(y_true: Sequence[float], y_pred: Sequence[float], strategy_returns: Sequence[float]) -> BacktestMetrics:
    """Compute core prediction and trading metrics."""
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    ret_arr = np.asarray(strategy_returns)

    mse = float(np.mean((true_arr - pred_arr) ** 2))
    directional_accuracy = float(np.mean(np.sign(true_arr) == np.sign(pred_arr)))
    equity_curve = np.cumprod(1.0 + ret_arr)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max

    return BacktestMetrics(
        mse=mse,
        directional_accuracy=directional_accuracy,
        cumulative_return=float(equity_curve[-1] - 1.0) if equity_curve.size > 0 else 0.0,
        max_drawdown=float(np.min(drawdown)) if drawdown.size > 0 else 0.0,
    )
