"""Tests for end-to-end dataset preparation."""

from __future__ import annotations

import numpy as np

from data_pipeline.indicators import FEATURE_COLUMNS
from data_pipeline.pipeline import prepare_datasets
from tests.conftest import make_raw_ohlcv


def test_prepare_datasets_shapes_and_order():
    data = prepare_datasets(make_raw_ohlcv(2000), sequence_length=32, train_split=0.7, val_split=0.15, timeframe="1h")

    for split in (data.train, data.val, data.test):
        assert split.X.ndim == 3 and split.X.shape[1:] == (32, len(FEATURE_COLUMNS))
        assert split.y.shape == (len(split), 1)
        assert len(split.timestamps) == len(split)
        assert not np.isnan(split.X).any() and not np.isnan(split.y).any()

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
