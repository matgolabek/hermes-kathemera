"""End-to-end preparation of model-ready datasets from raw OHLCV candles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from data_pipeline.indicators import FEATURE_COLUMNS, add_technical_indicators
from data_pipeline.preprocess import (
    CLASS_NAMES,
    FeatureScaler,
    add_target,
    class_names_for,
    clean_ohlcv,
    make_supervised_sequences,
    split_chronological,
)


@dataclass
class SequenceSet:
    """Model inputs and targets for one split.

    Attributes:
        X: Feature windows, float32 ``(samples, sequence_length, features)``.
        y: Class index per window (see ``class_names``), int64 ``(samples,)``; the
            format ``nn.CrossEntropyLoss`` expects.
        returns: Forward log return behind each label (over ``horizon`` candles),
            float32 ``(samples,)``.
        next_returns: Log return from each window's last close to the next close,
            float32 ``(samples,)``; what a position held for one candle earns.
        timestamps: Time of each window's last candle.
        class_names: Name of each class index.
    """

    X: np.ndarray
    y: np.ndarray
    returns: np.ndarray
    next_returns: np.ndarray
    timestamps: pd.DatetimeIndex
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))

    def __len__(self) -> int:
        return len(self.X)

    def tail(self, n: int) -> "SequenceSet":
        """The last ``n`` samples (the most recent windows)."""
        start = max(len(self) - n, 0)
        return SequenceSet(
            X=self.X[start:],
            y=self.y[start:],
            returns=self.returns[start:],
            next_returns=self.next_returns[start:],
            timestamps=self.timestamps[start:],
            class_names=list(self.class_names),
        )

    def class_counts(self) -> dict[str, int]:
        """Number of samples per class name."""
        counts = np.bincount(self.y, minlength=len(self.class_names))
        return {name: int(count) for name, count in zip(self.class_names, counts)}


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
    label_mode: str = "three_class"

    @property
    def class_names(self) -> list[str]:
        """Name of each class index for this label mode."""
        return class_names_for(self.label_mode)

    def metadata(self) -> dict[str, Any]:
        """Preprocessing settings to store with a model checkpoint."""
        return {
            "feature_columns": list(self.feature_columns),
            "scaler": self.scaler.to_dict(),
            "sequence_length": self.sequence_length,
            "horizon": self.horizon,
            "flat_threshold": self.flat_threshold,
            "label_mode": self.label_mode,
            "class_names": self.class_names,
        }


def build_features(
    raw: pd.DataFrame, timeframe: Optional[str] = None, feature_columns: Sequence[str] = FEATURE_COLUMNS
) -> pd.DataFrame:
    """Clean candles and add indicator features, dropping the indicator warm-up rows."""
    df = add_technical_indicators(clean_ohlcv(raw, timeframe))
    return df.dropna(subset=list(feature_columns))


def prepare_parts(
    parts: Sequence[pd.DataFrame],
    sequence_length: int,
    horizon: int,
    flat_threshold: float,
    label_mode: str = "three_class",
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    scaler: Optional[FeatureScaler] = None,
) -> PreparedData:
    """Label, scale and window three consecutive feature frames (train, val, test).

    Leakage guards:
    - Targets are computed per part, so none of them looks past the part's end; the
      last ``horizon`` rows of each part are dropped as window ends.
    - The scaler is fitted on the train part only (unless one is given).
    - Each part is windowed on its own rows, so no window crosses a boundary.
    """
    class_names = class_names_for(label_mode)
    labelled = [add_target(part, horizon, flat_threshold, label_mode) for part in parts]
    if scaler is None:
        scaler = FeatureScaler.fit(labelled[0], feature_columns)

    sets = []
    for part in labelled:
        X, targets, ends = make_supervised_sequences(
            scaler.transform(part),
            sequence_length,
            feature_columns,
            target_columns=("label", "target", "next_return"),
        )
        sets.append(
            SequenceSet(
                X=X,
                y=targets[:, 0].astype(np.int64),
                returns=targets[:, 1],
                next_returns=targets[:, 2],
                timestamps=ends,
                class_names=class_names,
            )
        )

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
        label_mode=label_mode,
    )


def prepare_datasets(
    raw: pd.DataFrame,
    sequence_length: int,
    train_split: float,
    val_split: float,
    timeframe: Optional[str] = None,
    horizon: int = 1,
    flat_threshold: float = 0.002,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    scaler: Optional[FeatureScaler] = None,
    label_mode: str = "three_class",
) -> PreparedData:
    """Clean candles, add features and labels, split by time, scale and window.

    With ``label_mode="three_class"`` each window is labelled up/flat/down by the
    return over the next ``horizon`` candles, with moves within ``±flat_threshold``
    counted as flat. With ``"binary"`` it is labelled down (a drop below
    ``-flat_threshold``) or rest.

    Pass ``scaler`` (e.g. restored from a checkpoint) to reuse it instead of fitting a
    new one on the training split. See ``prepare_parts`` for the leakage guards.
    """
    df = build_features(raw, timeframe, feature_columns)
    parts = split_chronological(df, train_split, val_split)
    return prepare_parts(parts, sequence_length, horizon, flat_threshold, label_mode, feature_columns, scaler)


def walk_forward_parts(
    df: pd.DataFrame, n_folds: int, min_train_fraction: float, val_fraction: float
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Split a feature frame into expanding-window walk-forward folds.

    The first ``min_train_fraction`` of rows is only ever used for fitting. The rest
    is cut into ``n_folds`` consecutive evaluation blocks. Fold ``k`` fits on all rows
    before its block, of which the last ``val_fraction`` is held out for early
    stopping, and is evaluated on block ``k``. Every evaluation block is therefore
    out-of-sample and later than everything its model has seen.

    Returns ``(train, val, eval)`` frames per fold.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if not 0 < min_train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("min_train_fraction and val_fraction must be between 0 and 1")

    n = len(df)
    first_eval = int(n * min_train_fraction)
    bounds = np.linspace(first_eval, n, n_folds + 1).astype(int)
    folds = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        val_start = int(start * (1 - val_fraction))
        folds.append((df.iloc[:val_start], df.iloc[val_start:start], df.iloc[start:end]))
    return folds


def prepare_walk_forward(
    raw: pd.DataFrame,
    n_folds: int,
    sequence_length: int,
    min_train_fraction: float = 0.5,
    val_fraction: float = 0.1,
    timeframe: Optional[str] = None,
    horizon: int = 1,
    flat_threshold: float = 0.002,
    label_mode: str = "three_class",
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> list[PreparedData]:
    """Build one ``PreparedData`` per walk-forward fold; ``test`` is the fold's evaluation block.

    Each fold has its own scaler fitted on its own training rows.
    """
    df = build_features(raw, timeframe, feature_columns)
    return [
        prepare_parts(parts, sequence_length, horizon, flat_threshold, label_mode, feature_columns)
        for parts in walk_forward_parts(df, n_folds, min_train_fraction, val_fraction)
    ]
