"""Data loader interfaces for OHLCV ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class OHLCVLoader:
    """Abstract-like loader interface for OHLCV data sources."""

    def load(self) -> pd.DataFrame:
        """Load OHLCV data as a DataFrame."""
        raise NotImplementedError


class CSVLoader(OHLCVLoader):
    """Loads OHLCV data from a CSV file."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """Load OHLCV rows from a CSV file."""
        return pd.read_csv(self.file_path)


class CCXTLoader(OHLCVLoader):
    """Fetches OHLCV data from an exchange via CCXT."""

    def __init__(self, exchange_name: str, symbol: str, timeframe: str, limit: int = 1000) -> None:
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit

    def load(self) -> pd.DataFrame:
        """Fetch OHLCV data from exchange via CCXT."""
        # TODO: Implement live market data fetch via CCXT.
        pass
