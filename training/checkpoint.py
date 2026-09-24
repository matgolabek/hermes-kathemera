"""Checkpoint save/load helpers for training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from config import ModelConfig
from models.factory import build_model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Persist model and optimizer state to disk.

    ``metadata`` should hold everything needed to use the model later without the
    training code: ``model_config`` (``dataclasses.asdict(ModelConfig)``) plus the
    preprocessing settings from ``PreparedData.metadata()`` (features, scaler, labels).
    It must contain only plain Python types so checkpoints load with ``weights_only``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> int:
    """Load model and optimizer state to resume training, and return the saved epoch."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"])


def load_model(path: Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a trained model from its checkpoint alone, in eval mode, with its metadata."""
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint.get("metadata") or {}
    if "model_config" not in metadata:
        raise ValueError(f"Checkpoint {path} has no model_config metadata; it cannot be rebuilt")
    model = build_model(ModelConfig(**metadata["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), metadata
