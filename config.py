"""Central configuration objects for training and execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and preprocessing."""

    exchange: str
    symbol: str
    timeframe: str
    since: str
    cache_dir: Path
    sequence_length: int
    train_split: float
    val_split: float


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for neural network architecture."""

    model_name: str
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training loop setup."""

    batch_size: int
    learning_rate: float
    epochs: int
    checkpoint_dir: Path
    device: str


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for live execution and risk constraints."""

    exchange_api_key: str
    exchange_api_secret: str
    stop_loss_pct: float
    order_size: float


def _get_env(name: str, default: str) -> str:
    """Read an environment variable with a string default."""
    return os.getenv(name, default)


def get_data_config() -> DataConfig:
    """Build data config from environment variables."""
    return DataConfig(
        exchange=_get_env("EXCHANGE", "binance"),
        symbol=_get_env("SYMBOL", "BTC/USDT"),
        timeframe=_get_env("TIMEFRAME", "1h"),
        since=_get_env("SINCE", "2020-01-01"),
        cache_dir=Path(_get_env("DATA_CACHE_DIR", "data/cache")),
        sequence_length=int(_get_env("SEQUENCE_LENGTH", "64")),
        train_split=float(_get_env("TRAIN_SPLIT", "0.8")),
        val_split=float(_get_env("VAL_SPLIT", "0.1")),
    )


def get_model_config() -> ModelConfig:
    """Build model config from environment variables."""
    return ModelConfig(
        model_name=_get_env("MODEL_NAME", "lstm"),
        input_size=int(_get_env("INPUT_SIZE", "16")),
        hidden_size=int(_get_env("HIDDEN_SIZE", "64")),
        num_layers=int(_get_env("NUM_LAYERS", "2")),
        output_size=int(_get_env("OUTPUT_SIZE", "1")),
        dropout=float(_get_env("DROPOUT", "0.1")),
    )


def get_training_config() -> TrainingConfig:
    """Build training config from environment variables."""
    return TrainingConfig(
        batch_size=int(_get_env("BATCH_SIZE", "32")),
        learning_rate=float(_get_env("LEARNING_RATE", "0.001")),
        epochs=int(_get_env("EPOCHS", "20")),
        checkpoint_dir=Path(_get_env("CHECKPOINT_DIR", "checkpoints")),
        device=_get_env("DEVICE", "cpu"),
    )


def get_execution_config() -> ExecutionConfig:
    """Build execution config from environment variables."""
    return ExecutionConfig(
        exchange_api_key=_get_env("EXCHANGE_API_KEY", ""),
        exchange_api_secret=_get_env("EXCHANGE_API_SECRET", ""),
        stop_loss_pct=float(_get_env("STOP_LOSS_PCT", "0.02")),
        order_size=float(_get_env("ORDER_SIZE", "0.001")),
    )
