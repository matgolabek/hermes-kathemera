"""End-to-end preparation of model-ready datasets from raw OHLCV candles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from data_pipeline.indicators import FEATURE_COLUMNS, add_technical_indicators
from data_pipeline.preprocess import (
    CLASS_NAMES,
    FeatureScaler,
    add_target,
    clean_ohlcv,
    make_supervised_sequences,
    split_chronological,
)


@dataclass
class SequenceSet:
    """Model inputs and targets for one split.

    Attributes:
        X: Feature windows, float32 ``(samples, sequence_length, features)``.
        y: Class index per window (``DOWN``/``FLAT``/``UP``), int64 ``(samples,)``;
            the format ``nn.CrossEntropyLoss`` expects.
        returns: Forward log return behind each label, float32 ``(samples,)``; used
            to measure trading results.
        timestamps: Time of each window's last candle.
    """

    X: np.ndarray
    y: np.ndarray
    returns: np.ndarray
    timestamps: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.X)

    def class_counts(self) -> dict[str, int]:
        """Number of samples per class name."""
        counts = np.bincount(self.y, minlength=len(CLASS_NAMES))
        return {name: int(count) for name, count in zip(CLASS_NAMES, counts)}


@dataclass
class PreparedData:
    """Train/validation/test sequences plus the settings needed to rebuild them live."""

    train: SequenceSet
    val: SequenceSet
    test: SequenceSet
    scaler: FeatureScaler
    feature_columns: list[str]
    sequence_length: int
    horizon: int
    flat_threshold: float

    def metadata(self) -> dict[str, Any]:
        """Preprocessing settings to store with a model checkpoint."""
        return {
            "feature_columns": list(self.feature_columns),
            "scaler": self.scaler.to_dict(),
            "sequence_length": self.sequence_length,
            "horizon": self.horizon,
            "flat_threshold": self.flat_threshold,
            "class_names": list(CLASS_NAMES),
        }


def prepare_datasets(
    raw: pd.DataFrame,
    sequence_length: int,
    train_split: float,
    val_split: float,
    timeframe: Optional[str] = None,
    horizon: int = 1,
    flat_threshold: float = 0.002,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> PreparedData:
    """Clean candles, add features and labels, split by time, scale and window.

    Each window is labelled up/flat/down by the return over the next ``horizon``
    candles, with moves within ``±flat_threshold`` counted as flat.

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
    parts = [add_target(part, horizon, flat_threshold) for part in parts]
    scaler = FeatureScaler.fit(parts[0], feature_columns)

    sets = []
    for part in parts:
        X, targets, ends = make_supervised_sequences(
            scaler.transform(part), sequence_length, feature_columns, target_columns=("label", "target")
        )
        sets.append(SequenceSet(X=X, y=targets[:, 0].astype(np.int64), returns=targets[:, 1], timestamps=ends))

    train, val, test = sets
    return PreparedData(
        train=train,
        val=val,
        test=test,
        scaler=scaler,
        feature_columns=list(feature_columns),
        sequence_length=sequence_length,
        horizon=horizon,
        flat_threshold=flat_threshold,
    )
