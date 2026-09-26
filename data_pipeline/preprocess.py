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

# Class names per label mode, in class-index order. "down" is class 0 in every mode, so
# column 0 of a model's probabilities is always P(down).
LABEL_MODES = {
    "three_class": ["down", "flat", "up"],
    "binary": ["down", "rest"],  # rest = flat or up: anything that is not a drop
}
CLASS_NAMES = LABEL_MODES["three_class"]
DOWN, FLAT, UP = range(len(CLASS_NAMES))
REST = 1


def class_names_for(label_mode: str) -> list[str]:
    """Class names of a label mode (``"three_class"`` or ``"binary"``)."""
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unknown label mode {label_mode!r}; expected one of {sorted(LABEL_MODES)}")
    return list(LABEL_MODES[label_mode])


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


def add_target(
    df: pd.DataFrame, horizon: int = 1, flat_threshold: float = 0.0, label_mode: str = "three_class"
) -> pd.DataFrame:
    """Add the forward return and its class.

    - ``target``: log return from this candle's close to the close ``horizon`` candles later.
    - ``next_return``: log return to the next candle's close, which is what a position
      held for one candle earns (equal to ``target`` when ``horizon`` is 1).
    - ``label``: ``UP`` if ``target > flat_threshold``, ``DOWN`` if ``target < -flat_threshold``,
      otherwise ``FLAT``. Set the threshold to about the round-trip trading cost, so
      ``FLAT`` means "a move too small to trade profitably". With ``label_mode="binary"``
      the label is ``DOWN`` if ``target < -flat_threshold``, otherwise ``REST``.

    The last ``horizon`` rows have no future close and get NaN in ``target`` and ``label``.
    """
    df = df.copy()
    target = np.log(df["close"].shift(-horizon) / df["close"])
    class_names_for(label_mode)  # validate
    if label_mode == "binary":
        label = pd.Series(float(REST), index=df.index)
    else:
        label = pd.Series(float(FLAT), index=df.index)
        label[target > flat_threshold] = UP
    label[target < -flat_threshold] = DOWN
    df["target"] = target
    df["next_return"] = np.log(df["close"].shift(-1) / df["close"])
    df["label"] = label.where(target.notna())
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

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Plain-dict form for saving alongside model checkpoints."""
        return {"mean": {k: float(v) for k, v in self.mean.items()}, "std": {k: float(v) for k, v in self.std.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float]]) -> "FeatureScaler":
        """Rebuild a scaler saved with ``to_dict``."""
        return cls(mean=pd.Series(data["mean"], dtype="float64"), std=pd.Series(data["std"], dtype="float64"))

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
    target_columns: Sequence[str] = ("target",),
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Turn a time-ordered frame into sliding windows for sequence models.

    Sample ``i`` holds the ``sequence_length`` rows ending at row ``t``; its targets are
    ``target_columns`` at row ``t`` (e.g. the return after ``t``). Rows with any NaN
    target are not used as window ends.

    Returns:
        ``X`` with shape ``(samples, sequence_length, features)``, ``y`` with shape
        ``(samples, len(target_columns))``, both float32, and the timestamp of each
        window's last row.
    """
    if len(df) < sequence_length:
        empty_x = np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32)
        empty_y = np.empty((0, len(target_columns)), dtype=np.float32)
        return empty_x, empty_y, pd.DatetimeIndex([], name=df.index.name)

    features = df[list(feature_columns)].to_numpy(dtype=np.float32)
    if np.isnan(features).any():
        raise ValueError("Feature columns contain NaN; drop indicator warm-up rows first")

    windows = sliding_window_view(features, sequence_length, axis=0).transpose(0, 2, 1)
    targets = df[list(target_columns)].to_numpy(dtype=np.float32)[sequence_length - 1 :]
    ends = df.index[sequence_length - 1 :]

    keep = ~np.isnan(targets).any(axis=1)
    X = np.ascontiguousarray(windows[keep])
    return X, targets[keep], ends[keep]
