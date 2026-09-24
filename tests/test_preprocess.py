"""Tests for OHLCV cleaning, splitting, scaling and windowing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.preprocess import (
    FeatureScaler,
    add_target,
    clean_ohlcv,
    make_supervised_sequences,
    split_chronological,
)


def test_clean_parses_sorts_and_dedupes(raw_ohlcv):
    messy = pd.concat([raw_ohlcv.iloc[::-1], raw_ohlcv.iloc[[5]]])

    df = clean_ohlcv(messy, "1h")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert len(df) == len(raw_ohlcv)
    assert df["is_filled"].sum() == 0


def test_clean_accepts_date_strings_and_uppercase_columns(raw_ohlcv):
    raw = raw_ohlcv.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True).astype(str)
    raw.columns = [c.upper() for c in raw.columns]

    df = clean_ohlcv(raw)

    assert len(df) == len(raw_ohlcv)
    assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_clean_fills_gaps_with_previous_close(raw_ohlcv):
    raw = raw_ohlcv.drop(index=[10, 11, 12])

    df = clean_ohlcv(raw, "1h")

    assert len(df) == len(raw_ohlcv)
    gap = df.iloc[10:13]
    prev_close = df["close"].iloc[9]
    assert gap["is_filled"].tolist() == [1, 1, 1]
    assert (gap[["open", "high", "low", "close"]] == prev_close).all().all()
    assert (gap["volume"] == 0).all()


def test_clean_drops_invalid_candles(raw_ohlcv):
    raw = raw_ohlcv.copy()
    raw.loc[3, "close"] = np.nan
    raw.loc[4, "low"] = raw.loc[4, "high"] * 2  # low above high
    raw.loc[5, "volume"] = -1
    raw.loc[6, "open"] = 0

    df = clean_ohlcv(raw, "1h")

    assert len(df) == len(raw_ohlcv)
    assert df["is_filled"].iloc[3:7].tolist() == [1, 1, 1, 1]


def test_clean_infers_timeframe(raw_ohlcv):
    df = clean_ohlcv(raw_ohlcv.drop(index=[20]))

    assert pd.infer_freq(df.index) == "h"


def test_clean_rejects_missing_columns(raw_ohlcv):
    with pytest.raises(ValueError, match="volume"):
        clean_ohlcv(raw_ohlcv.drop(columns="volume"))


def test_add_target_is_forward_log_return(raw_ohlcv):
    df = add_target(clean_ohlcv(raw_ohlcv, "1h"), horizon=2)

    expected = np.log(df["close"].iloc[2] / df["close"].iloc[0])
    assert df["target"].iloc[0] == pytest.approx(expected)
    assert df["target"].iloc[-2:].isna().all()


def test_split_is_chronological(raw_ohlcv):
    df = clean_ohlcv(raw_ohlcv, "1h")

    train, val, test = split_chronological(df, 0.8, 0.1)

    assert (len(train), len(val), len(test)) == (400, 50, 50)
    assert train.index[-1] < val.index[0] and val.index[-1] < test.index[0]


def test_split_rejects_bad_fractions(raw_ohlcv):
    with pytest.raises(ValueError):
        split_chronological(raw_ohlcv, 0.9, 0.2)


def test_scaler_uses_fit_statistics():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
    other = pd.DataFrame({"a": [4.0], "b": [6.0]})

    scaler = FeatureScaler.fit(train, ["a", "b"])

    assert scaler.transform(train)["a"].tolist() == [-1.0, 0.0, 1.0]
    assert scaler.transform(other)["a"].iloc[0] == pytest.approx(2.0)
    assert scaler.transform(other)["b"].iloc[0] == pytest.approx(1.0)  # constant column: centred only


def test_sequences_shape_and_alignment():
    index = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"f1": np.arange(10.0), "f2": np.arange(10.0) * 10, "target": np.arange(10.0) / 100}, index=index)
    df.loc[index[-1], "target"] = np.nan

    X, y, ends = make_supervised_sequences(df, 3, ["f1", "f2"])

    assert X.shape == (7, 3, 2) and y.shape == (7, 1)
    assert X.dtype == np.float32 and y.dtype == np.float32
    assert X[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert X[0, :, 1].tolist() == [0.0, 10.0, 20.0]
    assert y[0, 0] == pytest.approx(0.02)  # target of the window's last row
    assert ends[0] == index[2] and ends[-1] == index[8]


def test_sequences_short_input_is_empty():
    index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame({"f": [1.0, 2.0], "target": [0.1, 0.2]}, index=index)

    X, y, ends = make_supervised_sequences(df, 5, ["f"])

    assert X.shape == (0, 5, 1) and y.shape == (0, 1) and len(ends) == 0


def test_sequences_reject_nan_features():
    index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"f": [np.nan, 1.0, 2.0, 3.0, 4.0], "target": 0.0}, index=index)

    with pytest.raises(ValueError, match="NaN"):
        make_supervised_sequences(df, 2, ["f"])
