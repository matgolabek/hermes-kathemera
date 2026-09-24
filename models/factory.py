"""Factory for constructing sequence models by name."""

from __future__ import annotations

from torch import nn

from config import ModelConfig
from models.baselines import ConstantModel, LinearModel, MLPModel
from models.recurrent import RNN_CELLS, RecurrentModel

MODEL_NAMES = ["constant", "linear", "mlp", *RNN_CELLS]


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate a model from config. Every model maps ``(batch, seq_len, features)`` to ``(batch, output_size)``."""
    model_name = config.model_name.lower()
    if model_name in RNN_CELLS:
        return RecurrentModel(
            cell=model_name,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            dropout=config.dropout,
        )
    if model_name == "constant":
        return ConstantModel(output_size=config.output_size)
    if model_name == "linear":
        return LinearModel(
            sequence_length=config.sequence_length, input_size=config.input_size, output_size=config.output_size
        )
    if model_name == "mlp":
        return MLPModel(
            sequence_length=config.sequence_length,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model_name: {config.model_name!r}; expected one of {MODEL_NAMES}")
