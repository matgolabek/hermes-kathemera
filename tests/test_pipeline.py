"""Tests for end-to-end dataset preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.indicators import FEATURE_COLUMNS
from data_pipeline.pipeline import build_features, prepare_datasets, prepare_walk_forward, walk_forward_parts
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


def test_binary_label_mode():
    data = prepare_datasets(make_raw_ohlcv(800), 16, 0.7, 0.15, timeframe="1h", flat_threshold=0.005, label_mode="binary")

    assert data.class_names == ["down", "rest"]
    assert data.metadata()["label_mode"] == "binary"
    assert data.metadata()["class_names"] == ["down", "rest"]
    assert set(data.train.class_counts()) == {"down", "rest"}
    assert (data.train.y[data.train.returns < -0.005] == DOWN).all()
    assert (data.train.y[data.train.returns >= -0.005] == 1).all()


def test_walk_forward_parts_are_ordered_and_expanding():
    df = build_features(make_raw_ohlcv(1000), "1h")

    folds = walk_forward_parts(df, n_folds=4, min_train_fraction=0.5, val_fraction=0.2)

    assert len(folds) == 4
    eval_rows = sum(len(ev) for _, _, ev in folds)
    assert eval_rows == len(df) - int(len(df) * 0.5)
    for k, (train, val, ev) in enumerate(folds):
        assert train.index[-1] < val.index[0] and val.index[-1] < ev.index[0]
        assert train.index[0] == df.index[0]  # expanding window: always starts at the beginning
        if k:
            previous_eval = folds[k - 1][2]
            assert previous_eval.index[-1] < ev.index[0]
            assert len(train) > len(folds[k - 1][0])


def test_walk_forward_rejects_bad_settings():
    df = build_features(make_raw_ohlcv(300), "1h")
    with pytest.raises(ValueError):
        walk_forward_parts(df, n_folds=0, min_train_fraction=0.5, val_fraction=0.1)
    with pytest.raises(ValueError):
        walk_forward_parts(df, n_folds=2, min_train_fraction=1.0, val_fraction=0.1)


def test_prepare_walk_forward_fits_a_scaler_per_fold():
    folds = prepare_walk_forward(make_raw_ohlcv(1500), n_folds=3, sequence_length=16, timeframe="1h")

    assert len(folds) == 3
    assert folds[0].scaler.mean.iloc[0] != folds[2].scaler.mean.iloc[0]
    for data in folds:
        assert data.train.timestamps[-1] < data.val.timestamps[0] < data.test.timestamps[0]
