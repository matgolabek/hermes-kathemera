"""Main training and validation loop entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.checkpoint import save_checkpoint


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> float:
    """Run one epoch for training (optimizer set) or validation (optimizer None)."""
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        predictions = model(features)
        loss = loss_fn(predictions, targets)

        if training and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(dataloader), 1)


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
) -> None:
    """Train model and save best checkpoint by validation loss.

    ``metadata`` is stored in the checkpoint; see ``training.checkpoint.save_checkpoint``.
    """
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None, device=device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, checkpoint_path, metadata)

        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
