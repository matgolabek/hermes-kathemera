"""Classification metrics for up/flat/down predictions."""

from __future__ import annotations

from typing import Sequence

import numpy as np

_EPS = 1e-12


def class_prior(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Class frequencies, e.g. of the training labels."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def classification_metrics(
    probs: np.ndarray, labels: np.ndarray, train_prior: np.ndarray, class_names: Sequence[str]
) -> dict[str, float]:
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
    """
    num_classes = len(class_names)
    if len(labels) == 0:
        raise ValueError("Cannot compute metrics on an empty split")
    preds = probs.argmax(axis=1)

    log_loss = float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + _EPS)))
    prior_log_loss = float(-np.mean(np.log(train_prior[labels] + _EPS)))
    recalls = [np.mean(preds[labels == c] == c) for c in range(num_classes) if np.any(labels == c)]

    metrics = {
        "log_loss": log_loss,
        "prior_log_loss": prior_log_loss,
        "skill": 1.0 - log_loss / prior_log_loss if prior_log_loss > 0 else 0.0,
        "accuracy": float(np.mean(preds == labels)),
        "majority_accuracy": float(np.mean(labels == int(np.argmax(train_prior)))),
        "balanced_accuracy": float(np.mean(recalls)),
    }
    pred_share = np.bincount(preds, minlength=num_classes) / len(preds)
    metrics.update({f"pred_{name}": float(share) for name, share in zip(class_names, pred_share)})
    return metrics
