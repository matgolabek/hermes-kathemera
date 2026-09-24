"""Tests for model construction, output shapes and checkpoint round trips."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from config import ModelConfig
from models import MODEL_NAMES, build_model
from models.recurrent import RecurrentModel
from training.checkpoint import load_model, save_checkpoint


def make_config(model_name: str, **overrides) -> ModelConfig:
    values = dict(
        model_name=model_name,
        sequence_length=8,
        input_size=5,
        hidden_size=16,
        num_layers=2,
        output_size=3,
        dropout=0.1,
    )
    values.update(overrides)
    return ModelConfig(**values)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_maps_windows_to_class_logits(name):
    model = build_model(make_config(name))

    out = model(torch.randn(4, 8, 5))

    assert out.shape == (4, 3)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_trains_with_cross_entropy(name):
    torch.manual_seed(0)
    model = build_model(make_config(name))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    x = torch.randn(32, 8, 5)
    y = torch.randint(0, 3, (32,))

    losses = []
    for _ in range(20):
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_model_name_is_case_insensitive():
    assert isinstance(build_model(make_config("GRU")), RecurrentModel)


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="transformer"):
        build_model(make_config("transformer"))


def test_constant_model_learns_class_prior():
    model = build_model(make_config("constant"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    y = torch.tensor([2] * 60 + [0] * 30 + [1] * 10)
    x = torch.randn(len(y), 8, 5)

    for _ in range(300):
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probs = model(x[:1]).softmax(dim=1)[0]
    assert probs.tolist() == pytest.approx([0.3, 0.1, 0.6], abs=0.02)


def test_single_layer_rnn_has_no_inter_layer_dropout():
    model = build_model(make_config("lstm", num_layers=1, dropout=0.5))

    assert model.rnn.dropout == 0.0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_checkpoint_rebuilds_model_from_metadata(tmp_path, name):
    config = make_config(name)
    model = build_model(config).eval()
    optimizer = torch.optim.Adam(model.parameters())
    metadata = {"model_config": dataclasses.asdict(config), "feature_columns": ["a", "b", "c", "d", "e"]}
    path = tmp_path / "best.pt"

    save_checkpoint(model, optimizer, epoch=3, path=path, metadata=metadata)
    restored, restored_meta = load_model(path)

    x = torch.randn(2, 8, 5)
    assert torch.allclose(restored(x), model(x))
    assert not restored.training
    assert restored_meta == metadata


def test_checkpoint_without_metadata_cannot_be_rebuilt(tmp_path):
    model = build_model(make_config("linear"))
    path = tmp_path / "old.pt"
    save_checkpoint(model, torch.optim.Adam(model.parameters()), epoch=1, path=path)

    with pytest.raises(ValueError, match="model_config"):
        load_model(path)
