"""Turn model predictions into positions and simulate trading them with costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from backtest.metrics import BacktestMetrics, calculate_metrics
from config import BacktestConfig
from data_pipeline.preprocess import CLASS_NAMES

STRATEGY_RULES = ["argmax", "exit-on-down"]


@dataclass
class BacktestResult:
    """Per-candle simulation and its summary metrics.

    ``frame`` columns: ``position`` (-1, 0 or 1), ``asset_return`` (next-candle simple
    return), ``gross_return`` (before costs), ``cost``, ``net_return`` and ``equity``
    (growth of 1 unit, after costs).
    """

    frame: pd.DataFrame
    metrics: BacktestMetrics
    gross_metrics: BacktestMetrics
    down_prob_threshold: Optional[float] = None


def positions_from_probs(
    probs: np.ndarray,
    allow_short: bool = False,
    min_confidence: float = 0.0,
    class_names: Sequence[str] = CLASS_NAMES,
    rule: str = "argmax",
    down_prob_threshold: float = 0.5,
) -> np.ndarray:
    """Map class probabilities to positions (1 long, 0 out, -1 short).

    Rules:
    - ``"argmax"``: the most likely class decides. up or rest → long, down → short if
      shorting is allowed, otherwise out, flat → out. Predictions whose probability is
      below ``min_confidence`` are treated as out.
    - ``"exit-on-down"``: long by default; out (or short, if allowed) whenever
      P(down) > ``down_prob_threshold``. ``min_confidence`` is not used.
    """
    names = list(class_names)
    p_down = probs[:, names.index("down")]
    down_position = -1 if allow_short else 0

    if rule == "exit-on-down":
        positions = np.ones(len(probs), dtype=np.int8)
        positions[p_down > down_prob_threshold] = down_position
        return positions
    if rule != "argmax":
        raise ValueError(f"Unknown strategy rule {rule!r}; expected one of {STRATEGY_RULES}")

    predicted = probs.argmax(axis=1)
    positions = np.zeros(len(probs), dtype=np.int8)
    for name in ("up", "rest"):
        if name in names:
            positions[predicted == names.index(name)] = 1
    positions[predicted == names.index("down")] = down_position
    positions[probs.max(axis=1) < min_confidence] = 0
    return positions


def calibrate_down_threshold(probs: np.ndarray, class_names: Sequence[str], exit_share: float) -> float:
    """P(down) threshold that is exceeded by the top ``exit_share`` of ``probs``.

    Pass probabilities from data *before* the traded period; computing the threshold on
    the traded period itself would use its future.
    """
    if not 0 < exit_share < 1:
        raise ValueError("exit_share must be between 0 and 1")
    if len(probs) == 0:
        raise ValueError("Need calibration predictions to set the exit threshold")
    p_down = probs[:, list(class_names).index("down")]
    return float(np.quantile(p_down, 1.0 - exit_share))


def simulate(
    positions: np.ndarray,
    next_log_returns: np.ndarray,
    cost_per_trade: float,
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Simulate holding ``positions[t]`` from candle ``t``'s close to the next close.

    The position is assumed to be traded at candle ``t``'s close, right after the model
    sees it. Every change in position costs ``cost_per_trade`` per unit (a flip from
    long to short counts twice), and any position left open is closed at the end.
    Shorts are modelled as the negative of the asset return, ignoring borrow and
    funding costs.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != np.shape(next_log_returns):
        raise ValueError("positions and next_log_returns must have the same length")

    asset_return = np.expm1(np.asarray(next_log_returns, dtype=np.float64))
    turnover = np.abs(np.diff(positions, prepend=0.0))
    if len(turnover):
        turnover[-1] += abs(positions[-1])
    cost = turnover * cost_per_trade
    gross = positions * asset_return
    net = gross - cost

    return pd.DataFrame(
        {
            "position": positions.astype(np.int8),
            "asset_return": asset_return,
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": net,
            "equity": np.cumprod(1.0 + net),
        },
        index=timestamps,
    )


def run_backtest(
    positions: np.ndarray,
    next_log_returns: np.ndarray,
    config: BacktestConfig,
    periods: float,
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> BacktestResult:
    """Simulate ``positions`` with the configured costs and summarize the result."""
    frame = simulate(positions, next_log_returns, config.cost_per_trade, timestamps)
    trades = int(np.count_nonzero(frame["turnover"]))
    zero = np.zeros(len(frame))
    return BacktestResult(
        frame=frame,
        metrics=calculate_metrics(frame["net_return"].to_numpy(), frame["position"].to_numpy(), frame["cost"].to_numpy(), trades, periods),
        gross_metrics=calculate_metrics(frame["gross_return"].to_numpy(), frame["position"].to_numpy(), zero, trades, periods),
    )
