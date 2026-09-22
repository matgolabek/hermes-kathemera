"""Factory for constructing sequence models by name."""

from __future__ import annotations

from torch import nn

from config import ModelConfig
from models.gru_model import GRUModel
from models.lstm_model import LSTMModel


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate a sequence model from config."""
    model_name = config.model_name.lower()
    if model_name == "lstm":
        return LSTMModel(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            dropout=config.dropout,
        )
    if model_name == "gru":
        return GRUModel(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model_name: {config.model_name}")
