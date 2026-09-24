"""Simple reference models that the recurrent models have to beat.

If an LSTM/GRU does not outperform these on validation data, its extra complexity
is not finding anything useful.
"""

from __future__ import annotations

import torch
from torch import nn


class ConstantModel(nn.Module):
    """Ignores the input and learns a single output vector.

    For classification it learns the class frequencies, so it always predicts the most
    common class: the score any model gets without using the features at all.
    """

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the learned vector for every sample in the batch."""
        return self.bias.expand(x.shape[0], -1)


class LinearModel(nn.Module):
    """One linear layer over the whole flattened window (logistic regression for classification)."""

    def __init__(self, sequence_length: int, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(sequence_length * input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, seq_len, features)`` to ``(batch, output_size)``."""
        return self.linear(x.flatten(start_dim=1))


class MLPModel(nn.Module):
    """Feed-forward network with one hidden layer over the flattened window."""

    def __init__(self, sequence_length: int, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sequence_length * input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, seq_len, features)`` to ``(batch, output_size)``."""
        return self.net(x.flatten(start_dim=1))
