"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_raw_ohlcv(n: int = 500, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    """Random-walk hourly candles in the loader's raw format (timestamp in UTC ms)."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": timestamps.as_unit("ms").astype("int64"),
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": rng.uniform(1, 10, n),
        }
    )


@pytest.fixture
def raw_ohlcv() -> pd.DataFrame:
    return make_raw_ohlcv()
