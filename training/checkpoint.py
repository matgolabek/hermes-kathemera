"""Checkpoint save/load helpers for training runs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    """Persist model and optimizer state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> int:
    """Load model and optimizer state and return saved epoch."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"])
