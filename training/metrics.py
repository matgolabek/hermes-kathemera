"""Classification metrics for down/flat/up (or down/rest) predictions."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

_EPS = 1e-12


def class_prior(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Class frequencies, e.g. of the training labels."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; 0 when either input is constant."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def direction_score(probs: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """How bullish each prediction is: P(up) - P(down), or -P(down) without an up class."""
    names = list(class_names)
    score = -probs[:, names.index("down")]
    if "up" in names:
        score = score + probs[:, names.index("up")]
    return score


def move_score(probs: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """How strongly a large move is expected: 1 - P(flat), or P(down) without a flat class."""
    names = list(class_names)
    if "flat" in names:
        return 1.0 - probs[:, names.index("flat")]
    return probs[:, names.index("down")]


def direction_metrics(probs: np.ndarray, returns: np.ndarray, class_names: Sequence[str]) -> dict[str, Optional[float]]:
    """Separate what a model knows about direction from what it knows about volatility.

    - ``direction_ic``: rank correlation between ``direction_score`` and the forward
      return. Around 0 means no directional information; even 0.02-0.05 is useful.
    - ``volatility_ic``: rank correlation between ``move_score`` and the absolute forward
      return. High volatility IC with direction IC near 0 means the model predicts
      *when* the price moves, not *which way*.
    - ``ic_noise``: 2 / sqrt(samples); correlations smaller than this are likely luck.
    - ``direction_accuracy``: among predictions of up or down, the share where the
      return had that sign (``None`` if the model never predicts a direction).
    """
    names = list(class_names)
    returns = np.asarray(returns, dtype=np.float64)
    preds = probs.argmax(axis=1)

    directional = np.zeros(len(preds), dtype=bool)
    correct = np.zeros(len(preds), dtype=bool)
    for name, sign in (("up", 1.0), ("down", -1.0)):
        if name in names:
            chosen = preds == names.index(name)
            directional |= chosen
            correct |= chosen & (np.sign(returns) == sign)

    return {
        "direction_ic": rank_correlation(direction_score(probs, names), returns),
        "volatility_ic": rank_correlation(move_score(probs, names), np.abs(returns)),
        "ic_noise": 2.0 / np.sqrt(len(returns)),
        "direction_accuracy": float(correct[directional].mean()) if directional.any() else None,
    }


def classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    train_prior: np.ndarray,
    class_names: Sequence[str],
    returns: Optional[np.ndarray] = None,
) -> dict[str, Optional[float]]:
    """Score predicted class probabilities against true labels.

    Besides the usual metrics, compares the model with the no-information baseline
    that always predicts the training class frequencies ``train_prior``:

    - ``log_loss``: cross-entropy of the model (lower is better).
    - ``prior_log_loss``: cross-entropy of the baseline on the same labels.
    - ``skill``: ``1 - log_loss / prior_log_loss``. Above 0 means the model's
      probabilities beat the baseline; at or below 0 means it learned nothing useful.
    - ``accuracy`` and ``majority_accuracy`` (always guessing the most common training
      class), ``balanced_accuracy`` (mean per-class recall), and the share of
      predictions per class (``pred_<name>``).

    With the forward ``returns``, also adds ``direction_metrics``.
    """
    num_classes = len(class_names)
    if len(labels) == 0:
        raise ValueError("Cannot compute metrics on an empty split")
    preds = probs.argmax(axis=1)

    log_loss = float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + _EPS)))
    prior_log_loss = float(-np.mean(np.log(train_prior[labels] + _EPS)))
    recalls = [np.mean(preds[labels == c] == c) for c in range(num_classes) if np.any(labels == c)]

    metrics: dict[str, Optional[float]] = {
        "log_loss": log_loss,
        "prior_log_loss": prior_log_loss,
        "skill": 1.0 - log_loss / prior_log_loss if prior_log_loss > 0 else 0.0,
        "accuracy": float(np.mean(preds == labels)),
        "majority_accuracy": float(np.mean(labels == int(np.argmax(train_prior)))),
        "balanced_accuracy": float(np.mean(recalls)),
    }
    pred_share = np.bincount(preds, minlength=num_classes) / len(preds)
    metrics.update({f"pred_{name}": float(share) for name, share in zip(class_names, pred_share)})
    if returns is not None:
        metrics.update(direction_metrics(probs, returns, class_names))
    return metrics
