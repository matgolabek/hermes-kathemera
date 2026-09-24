"""Recurrent (LSTM/GRU) sequence model."""

from __future__ import annotations

import torch
from torch import nn

RNN_CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}


class RecurrentModel(nn.Module):
    """LSTM or GRU backbone whose last hidden state feeds a normalized, dropout-regularized linear head."""

    def __init__(
        self, cell: str, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float
    ) -> None:
        super().__init__()
        if cell not in RNN_CELLS:
            raise ValueError(f"Unsupported RNN cell {cell!r}; expected one of {sorted(RNN_CELLS)}")
        self.rnn = RNN_CELLS[cell](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, seq_len, features)`` to ``(batch, output_size)``."""
        outputs, _ = self.rnn(x)
        return self.head(outputs[:, -1, :])
