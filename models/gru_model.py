"""GRU model definition for sequence forecasting/classification."""

from __future__ import annotations

import torch
from torch import nn


class GRUModel(nn.Module):
    """Simple GRU backbone followed by a linear output head."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over a batch of sequences."""
        outputs, _ = self.gru(x)
        last_timestep = outputs[:, -1, :]
        return self.head(last_timestep)
