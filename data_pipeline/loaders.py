"""Data loader interfaces for OHLCV ingestion."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional, Union

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


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
    """Fetches OHLCV history from an exchange via CCXT, with pagination and a CSV cache.

    Exchanges cap how many candles one request returns, so history is fetched page by
    page. Downloaded candles are stored in ``cache_dir``; later calls only fetch candles
    that are missing from the cache. Only closed candles are returned and cached, since
    the still-forming candle changes until its period ends.

    Timestamps are candle open times in UTC milliseconds, as returned by CCXT.
    """

    def __init__(
        self,
        exchange_name: str,
        symbol: str,
        timeframe: str,
        limit: int = 1000,
        since: Optional[Union[str, int, pd.Timestamp]] = None,
        cache_dir: Optional[Path] = Path("data/cache"),
        max_retries: int = 5,
        exchange: Optional[Any] = None,
    ) -> None:
        """Create a loader.

        Args:
            exchange_name: CCXT exchange id, e.g. ``"binance"``.
            symbol: Market symbol, e.g. ``"BTC/USDT"``.
            timeframe: Candle size, e.g. ``"1h"``.
            limit: Candles requested per page.
            since: Start of history (date string, pandas Timestamp or UTC ms).
                ``None`` requests the earliest data the exchange provides.
            cache_dir: Directory for the CSV cache; ``None`` disables caching.
            max_retries: Attempts per page on network errors before giving up.
            exchange: Pre-built CCXT exchange instance (mainly for tests).
        """
        if exchange is None:
            if exchange_name not in ccxt.exchanges:
                raise ValueError(f"Unknown CCXT exchange: {exchange_name!r}")
            exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})

        self.exchange_name = exchange_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.since = since
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.exchange = exchange
        self.timeframe_ms = ccxt.Exchange.parse_timeframe(timeframe) * 1000

    @property
    def cache_path(self) -> Optional[Path]:
        """Path of the CSV cache file for this exchange, symbol and timeframe."""
        if self.cache_dir is None:
            return None
        safe_symbol = self.symbol.replace("/", "-").replace(":", "-")
        return Path(self.cache_dir) / f"{self.exchange_name}_{safe_symbol}_{self.timeframe}.csv"

    def load(self) -> pd.DataFrame:
        """Return closed candles from ``since`` until now, fetching only what is not cached."""
        start = self._since_ms()
        end = self._last_closed_candle_ms()
        cached = self._read_cache()

        frames = [cached]
        if cached.empty:
            frames.append(self._fetch_range(start, end))
        else:
            first, last = int(cached["timestamp"].iloc[0]), int(cached["timestamp"].iloc[-1])
            if self.since is not None and start < first:
                frames.append(self._fetch_range(start, first - self.timeframe_ms))
            frames.append(self._fetch_range(last + self.timeframe_ms, end))

        fetched = [frame for frame in frames[1:] if not frame.empty]
        df = cached
        if fetched:
            df = (
                pd.concat([cached, *fetched], ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            self._write_cache(df)

        return df[df["timestamp"] >= start].reset_index(drop=True)

    def _since_ms(self) -> int:
        """Convert ``since`` to UTC milliseconds (0 when unset)."""
        if self.since is None:
            return 0
        if isinstance(self.since, int):
            return self.since
        ts = pd.Timestamp(self.since)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return int(ts.value // 1_000_000)

    def _last_closed_candle_ms(self) -> int:
        """Open time of the most recent candle whose period has fully ended."""
        now = self.exchange.milliseconds()
        return (now // self.timeframe_ms) * self.timeframe_ms - self.timeframe_ms

    def _fetch_range(self, start: int, end: int) -> pd.DataFrame:
        """Fetch candles with open times in ``[start, end]``, one page at a time."""
        rows: list[list[float]] = []
        cursor = start
        while cursor <= end:
            page = [row for row in self._fetch_page(cursor) if cursor <= row[0] <= end]
            if not page:
                # No newer data, or the exchange ignored `since`; stop instead of looping.
                break
            rows.extend(page)
            cursor = int(page[-1][0]) + self.timeframe_ms
            logger.debug("Fetched %d candles up to %s", len(rows), pd.to_datetime(page[-1][0], unit="ms"))

        if rows:
            logger.info("Fetched %d candles for %s %s from %s", len(rows), self.symbol, self.timeframe, self.exchange_name)
        return self._to_frame(rows)

    def _fetch_page(self, since: int) -> list[list[float]]:
        """Fetch one page of candles, retrying with exponential backoff on network errors."""
        for attempt in range(self.max_retries):
            try:
                return self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=self.limit)
            except ccxt.NetworkError as exc:
                if attempt == self.max_retries - 1:
                    raise
                delay = 2**attempt
                logger.warning("fetch_ohlcv failed (%s); retrying in %ds", exc, delay)
                time.sleep(delay)
        return []

    def _read_cache(self) -> pd.DataFrame:
        """Load cached candles, or an empty frame when there is no cache."""
        path = self.cache_path
        if path is None or not path.exists():
            return self._to_frame([])
        df = pd.read_csv(path)
        missing = set(OHLCV_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Cache file {path} is missing columns: {sorted(missing)}")
        return self._to_frame(df[OHLCV_COLUMNS].values.tolist()).sort_values("timestamp").reset_index(drop=True)

    def _write_cache(self, df: pd.DataFrame) -> None:
        """Write candles to the cache via a temp file so a crash cannot corrupt it."""
        path = self.cache_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".csv.tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)

    @staticmethod
    def _to_frame(rows: list[list[float]]) -> pd.DataFrame:
        """Build an OHLCV DataFrame with integer timestamps and float prices."""
        df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
        return df.astype({"timestamp": "int64", **{col: "float64" for col in OHLCV_COLUMNS[1:]}})
