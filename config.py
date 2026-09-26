"""Central configuration objects for training and execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    horizon: int
    flat_threshold: float
    label_mode: str


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for neural network architecture."""

    model_name: str
    sequence_length: int
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
    weight_decay: float
    patience: int
    grad_clip: float
    seed: int


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for simulating trades from model predictions."""

    fee_rate: float
    slippage: float
    allow_short: bool
    min_confidence: float
    rule: str = "argmax"
    down_prob_threshold: float = 0.5
    exit_share: Optional[float] = None

    @property
    def cost_per_trade(self) -> float:
        """Cost of changing the position by one unit, as a fraction of the traded value."""
        return self.fee_rate + self.slippage


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
        horizon=int(_get_env("TARGET_HORIZON", "1")),
        flat_threshold=float(_get_env("FLAT_THRESHOLD", "0.002")),
        label_mode=_get_env("LABEL_MODE", "three_class"),
    )


def get_model_config() -> ModelConfig:
    """Build model config from environment variables."""
    return ModelConfig(
        model_name=_get_env("MODEL_NAME", "lstm"),
        sequence_length=int(_get_env("SEQUENCE_LENGTH", "64")),
        input_size=int(_get_env("INPUT_SIZE", "16")),
        hidden_size=int(_get_env("HIDDEN_SIZE", "64")),
        num_layers=int(_get_env("NUM_LAYERS", "2")),
        output_size=int(_get_env("OUTPUT_SIZE", "3")),
        dropout=float(_get_env("DROPOUT", "0.1")),
    )


def get_training_config() -> TrainingConfig:
    """Build training config from environment variables."""
    return TrainingConfig(
        batch_size=int(_get_env("BATCH_SIZE", "256")),
        learning_rate=float(_get_env("LEARNING_RATE", "0.001")),
        epochs=int(_get_env("EPOCHS", "20")),
        checkpoint_dir=Path(_get_env("CHECKPOINT_DIR", "checkpoints")),
        device=_get_env("DEVICE", "cpu"),
        weight_decay=float(_get_env("WEIGHT_DECAY", "0.0001")),
        patience=int(_get_env("PATIENCE", "5")),
        grad_clip=float(_get_env("GRAD_CLIP", "1.0")),
        seed=int(_get_env("SEED", "42")),
    )


def get_backtest_config() -> BacktestConfig:
    """Build backtest config from environment variables."""
    return BacktestConfig(
        fee_rate=float(_get_env("FEE_RATE", "0.001")),
        slippage=float(_get_env("SLIPPAGE", "0.0005")),
        allow_short=_get_env("ALLOW_SHORT", "false").lower() in {"1", "true", "yes"},
        min_confidence=float(_get_env("MIN_CONFIDENCE", "0.0")),
        rule=_get_env("STRATEGY_RULE", "argmax"),
        down_prob_threshold=float(_get_env("DOWN_PROB_THRESHOLD", "0.5")),
        exit_share=float(os.environ["EXIT_SHARE"]) if os.getenv("EXIT_SHARE") else None,
    )


def get_execution_config() -> ExecutionConfig:
    """Build execution config from environment variables."""
    return ExecutionConfig(
        exchange_api_key=_get_env("EXCHANGE_API_KEY", ""),
        exchange_api_secret=_get_env("EXCHANGE_API_SECRET", ""),
        stop_loss_pct=float(_get_env("STOP_LOSS_PCT", "0.02")),
        order_size=float(_get_env("ORDER_SIZE", "0.001")),
    )
