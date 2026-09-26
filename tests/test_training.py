"""Tests for the training loop, metrics and the model comparison runner."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import training.run as run
from config import ModelConfig, get_training_config
from data_pipeline.pipeline import PreparedData, SequenceSet
from data_pipeline.preprocess import CLASS_NAMES, DOWN, FLAT, UP, FeatureScaler
from models import build_model
from tests.conftest import make_raw_ohlcv
from training.metrics import class_prior, classification_metrics
from training.train import train_model


def learnable_split(n: int, seed: int, seq_len: int = 8, features: int = 4) -> SequenceSet:
    """Windows whose label is set by the sign of the last value of feature 0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, seq_len, features)).astype(np.float32)
    signal = X[:, -1, 0]
    y = np.where(signal > 0.4, UP, np.where(signal < -0.4, DOWN, FLAT)).astype(np.int64)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return SequenceSet(X=X, y=y, returns=signal / 100, next_returns=signal / 100, timestamps=timestamps)


def learnable_data() -> PreparedData:
    columns = ["f0", "f1", "f2", "f3"]
    scaler = FeatureScaler(mean=pd.Series(0.0, index=columns), std=pd.Series(1.0, index=columns))
    return PreparedData(
        train=learnable_split(2000, 0),
        val=learnable_split(400, 1),
        test=learnable_split(400, 2),
        scaler=scaler,
        feature_columns=columns,
        sequence_length=8,
        horizon=1,
        flat_threshold=0.002,
    )


def small_configs(tmp_path, **train_overrides):
    model_cfg = ModelConfig(
        model_name="linear", sequence_length=8, input_size=4, hidden_size=16, num_layers=1, output_size=3, dropout=0.0
    )
    settings = dict(checkpoint_dir=tmp_path, epochs=15, batch_size=64, learning_rate=0.01)
    train_cfg = dataclasses.replace(get_training_config(), **{**settings, **train_overrides})
    return model_cfg, train_cfg


def test_metrics_skill_is_zero_for_prior_and_high_for_perfect():
    labels = np.array([0, 0, 1, 2, 2, 2])
    prior = class_prior(labels, 3)

    as_prior = classification_metrics(np.tile(prior, (6, 1)), labels, prior, CLASS_NAMES)
    perfect = classification_metrics(np.eye(3)[labels], labels, prior, CLASS_NAMES)

    assert as_prior["skill"] == pytest.approx(0.0)
    assert perfect["skill"] == pytest.approx(1.0)
    assert perfect["accuracy"] == perfect["balanced_accuracy"] == 1.0
    assert as_prior["majority_accuracy"] == pytest.approx(0.5)
    assert as_prior["pred_up"] == 1.0  # argmax of the prior is the most common class


def test_direction_metrics_separate_direction_from_volatility():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 2000)
    names = ["down", "flat", "up"]

    # Knows the direction: P(up) grows with the return.
    bullish = 1 / (1 + np.exp(-returns * 300))
    knows_direction = np.column_stack([(1 - bullish) * 0.8, np.full(2000, 0.2), bullish * 0.8])
    # Knows only the size of the move: P(flat) falls as |return| grows, up and down equal.
    move = np.clip(np.abs(returns) * 50, 0, 0.9)
    knows_volatility = np.column_stack([move / 2, 1 - move, move / 2])

    direction = classification_metrics(knows_direction, np.zeros(2000, dtype=int), np.ones(3) / 3, names, returns)
    volatility = classification_metrics(knows_volatility, np.zeros(2000, dtype=int), np.ones(3) / 3, names, returns)

    assert direction["direction_ic"] > 0.9
    assert abs(volatility["direction_ic"]) < volatility["ic_noise"]
    assert volatility["volatility_ic"] > 0.9
    assert direction["direction_accuracy"] > 0.9


def test_direction_metrics_for_binary_and_no_directional_predictions():
    returns = np.array([-0.02, -0.01, 0.01, 0.02])
    p_down = np.array([0.4, 0.3, 0.2, 0.1])
    probs = np.column_stack([p_down, 1 - p_down])

    metrics = classification_metrics(probs, np.array([0, 0, 1, 1]), np.array([0.5, 0.5]), ["down", "rest"], returns)

    assert metrics["direction_ic"] == pytest.approx(1.0)  # lower P(down) ↔ higher return
    assert metrics["direction_accuracy"] is None  # argmax never picks "down"


def test_metrics_reject_empty_split():
    with pytest.raises(ValueError):
        classification_metrics(np.empty((0, 3)), np.empty(0, dtype=np.int64), np.ones(3) / 3, CLASS_NAMES)


def test_early_stopping_when_validation_does_not_improve(tmp_path):
    data = learnable_data()
    model = build_model(small_configs(tmp_path)[0])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # nothing changes, so no epoch beats the first

    history = train_model(
        model,
        run.make_loader(data.train, 64, shuffle=True),
        run.make_loader(data.val, 64, shuffle=False),
        nn.CrossEntropyLoss(),
        optimizer,
        epochs=20,
        device="cpu",
        checkpoint_path=tmp_path / "m.pt",
        patience=3,
    )

    assert history.best_epoch == 1
    assert len(history.val) == 4
    assert history.stopped_early
    assert (tmp_path / "m.pt").exists()


@pytest.mark.parametrize("name", ["linear", "mlp", "gru"])
def test_models_learn_a_real_signal(tmp_path, name):
    model_cfg, train_cfg = small_configs(tmp_path)

    result = run.train_and_evaluate(learnable_data(), dataclasses.replace(model_cfg, model_name=name), train_cfg)

    assert result["val"]["skill"] > 0.3
    assert result["val"]["accuracy"] > 0.8
    assert "test" not in result


def test_training_is_reproducible(tmp_path):
    model_cfg, train_cfg = small_configs(tmp_path, epochs=3)

    first = run.train_and_evaluate(learnable_data(), model_cfg, train_cfg)
    second = run.train_and_evaluate(learnable_data(), model_cfg, train_cfg)

    assert first["val"]["log_loss"] == pytest.approx(second["val"]["log_loss"])


def test_checkpoint_metadata_describes_the_run(tmp_path):
    model_cfg, train_cfg = small_configs(tmp_path, epochs=2)
    run.train_and_evaluate(learnable_data(), model_cfg, train_cfg)

    checkpoint = torch.load(tmp_path / "linear.pt")
    meta = checkpoint["metadata"]

    assert meta["model_config"]["model_name"] == "linear"
    assert meta["feature_columns"] == ["f0", "f1", "f2", "f3"]
    assert meta["best_epoch"] == checkpoint["epoch"]
    assert meta["training"]["seed"] == train_cfg.seed


def test_cli_runs_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run.CCXTLoader, "load", lambda self: make_raw_ohlcv(1500))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("SEQUENCE_LENGTH", "16")

    results = run.main(["--models", "linear", "gru", "--eval-test"])

    out = capsys.readouterr().out
    assert "[val]" in out and "[test]" in out
    assert [r["model"] for r in results] == ["linear", "gru"]
    assert all("test" in r for r in results)
    assert json.loads((tmp_path / "results.json").read_text())[0]["model"] == "linear"
    assert (tmp_path / "gru.pt").exists()


def test_cli_trains_binary_label_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run.CCXTLoader, "load", lambda self: make_raw_ohlcv(1500))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("SEQUENCE_LENGTH", "16")
    monkeypatch.setenv("LABEL_MODE", "binary")
    monkeypatch.setenv("FLAT_THRESHOLD", "0.005")

    results = run.main(["--models", "gru"])

    assert "predicted down/rest" in capsys.readouterr().out
    assert results[0]["class_names"] == ["down", "rest"]
    assert torch.load(tmp_path / "gru.pt")["metadata"]["label_mode"] == "binary"
