"""PyTorch Dataset definitions for sequence training."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset wrapping feature sequences and target values."""

    def __init__(self, features: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (feature, target) pair."""
        return self.features[index], self.targets[index]
