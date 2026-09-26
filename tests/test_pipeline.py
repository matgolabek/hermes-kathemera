"""Tests for end-to-end dataset preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.indicators import FEATURE_COLUMNS
from data_pipeline.pipeline import prepare_datasets
from data_pipeline.preprocess import DOWN, FLAT, UP
from tests.conftest import make_raw_ohlcv


def test_prepare_datasets_shapes_and_order():
    data = prepare_datasets(make_raw_ohlcv(2000), sequence_length=32, train_split=0.7, val_split=0.15, timeframe="1h")

    for split in (data.train, data.val, data.test):
        assert split.X.ndim == 3 and split.X.shape[1:] == (32, len(FEATURE_COLUMNS))
        assert split.y.shape == (len(split),) and split.y.dtype == np.int64
        assert split.returns.shape == (len(split),)
        assert len(split.timestamps) == len(split)
        assert not np.isnan(split.X).any() and not np.isnan(split.returns).any()
        assert set(np.unique(split.y)) <= {DOWN, FLAT, UP}

    assert data.train.timestamps[-1] < data.val.timestamps[0]
    assert data.val.timestamps[-1] < data.test.timestamps[0]
    assert data.feature_columns == FEATURE_COLUMNS


def test_train_features_are_standardized():
    data = prepare_datasets(make_raw_ohlcv(2000), sequence_length=16, train_split=0.7, val_split=0.15, timeframe="1h")

    # Last timestep of every window covers each training row once (after the first 15).
    last_step = data.train.X[:, -1, :]
    assert np.allclose(last_step.mean(axis=0), 0, atol=0.1)
    assert np.allclose(last_step.std(axis=0), 1, atol=0.1)


def test_targets_do_not_cross_split_boundaries():
    horizon = 3
    data = prepare_datasets(make_raw_ohlcv(1000), 16, 0.7, 0.15, timeframe="1h", horizon=horizon)

    # The last training window must end at least `horizon` candles before validation data.
    gap_hours = (data.val.timestamps[0] - data.train.timestamps[-1]).total_seconds() / 3600
    assert gap_hours >= horizon


def test_labels_match_returns_and_threshold():
    data = prepare_datasets(make_raw_ohlcv(1000), 16, 0.7, 0.15, timeframe="1h", flat_threshold=0.005)

    split = data.train
    assert (split.y[split.returns > 0.005] == UP).all()
    assert (split.y[split.returns < -0.005] == DOWN).all()
    assert (split.y[np.abs(split.returns) <= 0.005] == FLAT).all()
    assert sum(split.class_counts().values()) == len(split)


def test_metadata_is_plain_and_complete():
    data = prepare_datasets(make_raw_ohlcv(500), 16, 0.7, 0.15, timeframe="1h", horizon=2, flat_threshold=0.001)

    meta = data.metadata()

    assert meta["feature_columns"] == FEATURE_COLUMNS
    assert set(meta["scaler"]["mean"]) == set(FEATURE_COLUMNS)
    assert (meta["sequence_length"], meta["horizon"], meta["flat_threshold"]) == (16, 2, 0.001)
    assert meta["class_names"] == ["down", "flat", "up"]


def test_next_returns_are_one_candle_returns():
    raw = make_raw_ohlcv(600)
    data = prepare_datasets(raw, 16, 0.7, 0.15, timeframe="1h", horizon=3)

    close = pd.Series(raw["close"].to_numpy(), index=pd.to_datetime(raw["timestamp"], unit="ms", utc=True))
    t = data.val.timestamps[5]
    expected = np.log(close.shift(-1)[t] / close[t])
    assert data.val.next_returns[5] == pytest.approx(expected, rel=1e-5)
    # Consecutive windows sit on consecutive candles, so trades can be simulated in order.
    assert (np.diff(data.val.timestamps) == pd.Timedelta("1h")).all()


def test_prepare_datasets_reuses_given_scaler():
    raw = make_raw_ohlcv(600)
    first = prepare_datasets(raw, 16, 0.7, 0.15, timeframe="1h")

    second = prepare_datasets(make_raw_ohlcv(600, seed=1), 16, 0.7, 0.15, timeframe="1h", scaler=first.scaler)

    assert second.scaler is first.scaler
