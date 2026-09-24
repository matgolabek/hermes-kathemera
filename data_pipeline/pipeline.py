"""End-to-end preparation of model-ready datasets from raw OHLCV candles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from data_pipeline.indicators import FEATURE_COLUMNS, add_technical_indicators
from data_pipeline.preprocess import (
    FeatureScaler,
    add_target,
    clean_ohlcv,
    make_supervised_sequences,
    split_chronological,
)


@dataclass
class SequenceSet:
    """Model inputs, targets and the timestamp of each window's last candle."""

    X: np.ndarray
    y: np.ndarray
    timestamps: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.X)


@dataclass
class PreparedData:
    """Train/validation/test sequences plus the scaler fitted on the training period."""

    train: SequenceSet
    val: SequenceSet
    test: SequenceSet
    scaler: FeatureScaler
    feature_columns: list[str]


def prepare_datasets(
    raw: pd.DataFrame,
    sequence_length: int,
    train_split: float,
    val_split: float,
    timeframe: Optional[str] = None,
    horizon: int = 1,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> PreparedData:
    """Clean candles, add features and target, split by time, scale and window.

    Leakage guards:
    - Data is split by time before scaling; the scaler only sees training rows.
    - Each split is windowed on its own rows, so no window crosses a split boundary.
    - The last ``horizon`` rows of each split are dropped as window ends, since their
      target would need prices from the next split.
    """
    df = add_technical_indicators(clean_ohlcv(raw, timeframe))
    df = df.dropna(subset=list(feature_columns))

    parts = split_chronological(df, train_split, val_split)
    # Targets are computed per split so none of them looks past the split's end.
    parts = [add_target(part, horizon) for part in parts]
    scaler = FeatureScaler.fit(parts[0], feature_columns)

    train, val, test = (
        SequenceSet(*make_supervised_sequences(scaler.transform(part), sequence_length, feature_columns))
        for part in parts
    )
    return PreparedData(train=train, val=val, test=test, scaler=scaler, feature_columns=list(feature_columns))
