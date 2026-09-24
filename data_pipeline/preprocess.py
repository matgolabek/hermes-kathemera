"""Preprocessing utilities for time-series modeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import ccxt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close"]
OHLCV_COLUMNS = [*PRICE_COLUMNS, "volume"]


def clean_ohlcv(df: pd.DataFrame, timeframe: Optional[str] = None) -> pd.DataFrame:
    """Clean raw OHLCV rows into a gap-free, time-indexed frame.

    Steps: parse timestamps to a UTC ``DatetimeIndex``, coerce values to numbers,
    drop duplicate timestamps (keeping the last), drop invalid candles (missing or
    non-positive prices, negative volume, high/low not bounding open/close), sort by
    time, and fill missing candles.

    A missing candle usually means no trades or exchange downtime. It is filled with
    the previous close for all prices and zero volume, and marked ``is_filled = 1``.

    Args:
        df: Rows with ``timestamp`` (UTC ms or date strings) and OHLCV columns.
        timeframe: Candle size such as ``"1h"``; inferred from the data when omitted.
    """
    df = df.rename(columns=str.lower)
    missing = {"timestamp", *OHLCV_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")

    df = df[["timestamp", *OHLCV_COLUMNS]].copy()
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")

    df = df.drop_duplicates(subset="timestamp", keep="last").set_index("timestamp").sort_index()

    valid = (
        df[OHLCV_COLUMNS].notna().all(axis=1)
        & (df[PRICE_COLUMNS] > 0).all(axis=1)
        & (df["volume"] >= 0)
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
    )
    if not valid.all():
        logger.warning("Dropping %d invalid candles", int((~valid).sum()))
    df = df[valid]
    if df.empty:
        raise ValueError("No valid candles left after cleaning")

    freq = pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(timeframe)) if timeframe else _infer_freq(df.index)
    full_index = pd.date_range(df.index[0], df.index[-1], freq=freq, name="timestamp")
    off_grid = df.index.difference(full_index)
    if len(off_grid):
        logger.warning("Dropping %d candles not aligned to the %s grid", len(off_grid), freq)

    df = df.reindex(full_index)
    filled = df["close"].isna()
    if filled.any():
        logger.info("Filling %d missing candles (%.2f%%)", int(filled.sum()), 100 * filled.mean())
    df["close"] = df["close"].ffill()
    for col in ["open", "high", "low"]:
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df["is_filled"] = filled.astype("int8")
    return df


def _infer_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Return the most common spacing between consecutive timestamps."""
    if len(index) < 2:
        raise ValueError("Need a timeframe or at least two candles to infer it")
    return pd.Series(index).diff().dropna().mode().iloc[0]


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add ``target``: the log return from this candle's close to the close ``horizon`` candles later.

    The last ``horizon`` rows have no future close and get NaN.
    """
    df = df.copy()
    df["target"] = np.log(df["close"].shift(-horizon) / df["close"])
    return df


def split_chronological(df: pd.DataFrame, train_split: float, val_split: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by time into train, validation and test parts (oldest first, never shuffled)."""
    if not 0 < train_split < 1 or not 0 <= val_split < 1 or train_split + val_split >= 1:
        raise ValueError("Need 0 < train_split, 0 <= val_split and train_split + val_split < 1")
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


@dataclass
class FeatureScaler:
    """Standardizes feature columns with statistics from the training period only."""

    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: Sequence[str]) -> "FeatureScaler":
        """Compute per-column mean and standard deviation."""
        std = df[list(columns)].std()
        # Constant columns would divide by zero; leave them centred but unscaled.
        return cls(mean=df[list(columns)].mean(), std=std.where(std > 0, 1.0))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with the fitted columns standardized."""
        df = df.copy()
        columns = list(self.mean.index)
        df[columns] = (df[columns] - self.mean) / self.std
        return df


def make_supervised_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: Sequence[str],
    target_column: str = "target",
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Turn a time-ordered frame into sliding windows for sequence models.

    Sample ``i`` holds the ``sequence_length`` rows ending at row ``t``; its target is
    ``target_column`` at row ``t`` (e.g. the return after ``t``). Rows whose target is
    NaN are not used as window ends.

    Returns:
        ``X`` with shape ``(samples, sequence_length, features)``, ``y`` with shape
        ``(samples, 1)`` (matching a model with ``output_size=1``), both float32, and
        the timestamp of each window's last row.
    """
    if len(df) < sequence_length:
        empty_x = np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32)
        return empty_x, np.empty((0, 1), dtype=np.float32), pd.DatetimeIndex([], name=df.index.name)

    features = df[list(feature_columns)].to_numpy(dtype=np.float32)
    if np.isnan(features).any():
        raise ValueError("Feature columns contain NaN; drop indicator warm-up rows first")

    windows = sliding_window_view(features, sequence_length, axis=0).transpose(0, 2, 1)
    targets = df[target_column].to_numpy(dtype=np.float32)[sequence_length - 1 :]
    ends = df.index[sequence_length - 1 :]

    keep = ~np.isnan(targets)
    X = np.ascontiguousarray(windows[keep])
    y = targets[keep].reshape(-1, 1)
    return X, y, ends[keep]
