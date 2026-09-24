"""Tests for technical indicator features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.indicators import FEATURE_COLUMNS, add_technical_indicators
from data_pipeline.preprocess import clean_ohlcv


def test_adds_all_features_without_nan_after_warmup(raw_ohlcv):
    df = add_technical_indicators(clean_ohlcv(raw_ohlcv, "1h"))

    assert set(FEATURE_COLUMNS) <= set(df.columns)
    after_warmup = df[FEATURE_COLUMNS].iloc[60:]
    assert not after_warmup.isna().any().any()
    assert np.isfinite(after_warmup.to_numpy()).all()


def test_features_do_not_use_future_data(raw_ohlcv):
    base = add_technical_indicators(clean_ohlcv(raw_ohlcv, "1h"))

    altered = raw_ohlcv.copy()
    cut = 300
    altered.loc[cut:, ["open", "high", "low", "close"]] *= 3
    altered.loc[cut:, "volume"] *= 10
    changed = add_technical_indicators(clean_ohlcv(altered, "1h"))

    pd.testing.assert_frame_equal(base[FEATURE_COLUMNS].iloc[:cut], changed[FEATURE_COLUMNS].iloc[:cut])


def test_rsi_bounds_and_flat_periods_are_finite(raw_ohlcv):
    raw = raw_ohlcv.drop(index=range(100, 160))  # 60 filled, flat candles
    df = add_technical_indicators(clean_ohlcv(raw, "1h"))

    rsi = df["rsi_14"].dropna()
    assert rsi.between(0, 1).all()
    assert not df[FEATURE_COLUMNS].iloc[60:].isna().any().any()
    assert df["bb_pctb_20"].iloc[150] == 0.5
    assert df["volume_z_24"].iloc[150] == 0.0


def test_calendar_features():
    index = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")  # Monday
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=index)

    out = add_technical_indicators(df)

    assert out["hour_sin"].iloc[6] == 1.0  # 06:00 is a quarter of the day
    assert out["dow_sin"].iloc[0] == 0.0 and out["dow_cos"].iloc[0] == 1.0
