"""Main training and validation loop entry points."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    """Mean loss and accuracy over one pass of a dataloader."""

    loss: float
    accuracy: float


@dataclass
class TrainingHistory:
    """Per-epoch results and the epoch whose checkpoint was kept."""

    train: list[EpochResult] = field(default_factory=list)
    val: list[EpochResult] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    grad_clip: Optional[float] = None,
) -> EpochResult:
    """Run one epoch for training (optimizer set) or validation (optimizer None).

    Loss and accuracy are averaged per sample, so a smaller last batch does not skew them.
    """
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    correct = 0
    seen = 0

    with torch.set_grad_enabled(training):
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)
            loss = loss_fn(predictions, targets)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch = len(targets)
            total_loss += float(loss.item()) * batch
            correct += int((predictions.argmax(dim=1) == targets).sum().item())
            seen += batch

    return EpochResult(loss=total_loss / max(seen, 1), accuracy=correct / max(seen, 1))


@torch.no_grad()
def predict_proba(model: nn.Module, dataloader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return softmax class probabilities and the true labels for every sample."""
    model.eval()
    probs, labels = [], []
    for features, targets in dataloader:
        probs.append(model(features.to(device)).softmax(dim=1).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    checkpoint_path: Path,
    metadata: Optional[dict[str, Any]] = None,
    patience: Optional[int] = None,
    grad_clip: Optional[float] = None,
) -> TrainingHistory:
    """Train model and save best checkpoint by validation loss.

    Stops early once validation loss has not improved for ``patience`` epochs.
    ``metadata`` is stored in the checkpoint; see ``training.checkpoint.save_checkpoint``.
    """
    model.to(device)
    history = TrainingHistory()

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip)
        val_result = run_epoch(model, val_loader, loss_fn, optimizer=None, device=device)
        history.train.append(train_result)
        history.val.append(val_result)

        improved = val_result.loss < history.best_val_loss
        if improved:
            history.best_val_loss = val_result.loss
            history.best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, checkpoint_path, {**(metadata or {}), "best_epoch": epoch})

        logger.info(
            "epoch=%d train_loss=%.5f train_acc=%.3f val_loss=%.5f val_acc=%.3f%s",
            epoch,
            train_result.loss,
            train_result.accuracy,
            val_result.loss,
            val_result.accuracy,
            " *" if improved else "",
        )

        if patience is not None and epoch - history.best_epoch >= patience:
            history.stopped_early = True
            logger.info("Early stop: no validation improvement for %d epochs", patience)
            break

    return history
