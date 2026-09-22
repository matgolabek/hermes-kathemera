"""Preprocessing utilities for time-series modeling."""

from __future__ import annotations

import pandas as pd


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize raw OHLCV rows."""
    # TODO: Handle missing values, timestamp parsing, duplicates, and sorting.
    pass


def make_supervised_sequences(df: pd.DataFrame, sequence_length: int) -> tuple[pd.DataFrame, pd.Series]:
    """Transform a time-series DataFrame into supervised sequences."""
    # TODO: Build rolling windows and aligned targets.
    pass
