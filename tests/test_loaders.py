"""Tests for CCXTLoader pagination and caching, using a simulated exchange."""

from __future__ import annotations

import ccxt
import pandas as pd
import pytest

from data_pipeline.loaders import OHLCV_COLUMNS, CCXTLoader

HOUR_MS = 3_600_000


class FakeExchange:
    """Minimal stand-in for a CCXT exchange serving hourly candles."""

    def __init__(self, first_ms: int, now_ms: int, max_page: int = 1000, failures: int = 0) -> None:
        self.first_ms = first_ms
        self.now_ms = now_ms
        self.max_page = max_page
        self.failures = failures
        self.calls: list[int] = []

    def milliseconds(self) -> int:
        return self.now_ms

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
        if self.failures:
            self.failures -= 1
            raise ccxt.NetworkError("temporary outage")
        self.calls.append(since)
        start = max(since, self.first_ms)
        start += -start % HOUR_MS
        rows = []
        ts = start
        # Like real exchanges, includes the still-forming candle.
        while ts <= self.now_ms and len(rows) < min(limit, self.max_page):
            rows.append([ts, 1.0, 2.0, 0.5, 1.5, 10.0])
            ts += HOUR_MS
        return rows


# Exchange "now" values on an hour boundary mean candle N-1 has just closed,
# so N closed candles (0..N-1) are available.


def make_loader(exchange: FakeExchange, tmp_path, **kwargs) -> CCXTLoader:
    return CCXTLoader("binance", "BTC/USDT", "1h", exchange=exchange, cache_dir=tmp_path, **kwargs)


def test_paginates_and_excludes_open_candle(tmp_path):
    now = 2500 * HOUR_MS + HOUR_MS // 2  # halfway through candle 2500
    exchange = FakeExchange(first_ms=0, now_ms=now)

    df = make_loader(exchange, tmp_path).load()

    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 2500  # candles 0..2499; candle 2500 is still open
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].diff().dropna().eq(HOUR_MS).all()
    assert len(exchange.calls) == 3  # 1000 + 1000 + 500


def test_respects_exchange_page_cap(tmp_path):
    exchange = FakeExchange(first_ms=0, now_ms=1200 * HOUR_MS, max_page=500)

    df = make_loader(exchange, tmp_path).load()

    assert len(df) == 1200
    assert len(exchange.calls) == 3


def test_cache_is_written_and_only_new_candles_fetched(tmp_path):
    exchange = FakeExchange(first_ms=0, now_ms=100 * HOUR_MS)
    loader = make_loader(exchange, tmp_path)
    loader.load()
    assert loader.cache_path == tmp_path / "binance_BTC-USDT_1h.csv"
    assert len(pd.read_csv(loader.cache_path)) == 100

    exchange.now_ms = 150 * HOUR_MS
    exchange.calls.clear()
    df = loader.load()

    assert len(df) == 150
    assert exchange.calls == [100 * HOUR_MS]
    assert len(pd.read_csv(loader.cache_path)) == 150


def test_up_to_date_cache_needs_no_download_beyond_check(tmp_path):
    exchange = FakeExchange(first_ms=0, now_ms=100 * HOUR_MS)
    loader = make_loader(exchange, tmp_path)
    loader.load()
    exchange.calls.clear()

    df = loader.load()

    assert len(df) == 100
    assert exchange.calls == []  # next candle is not closed yet, so nothing to fetch


def test_since_filters_and_extends_cache_backwards(tmp_path):
    exchange = FakeExchange(first_ms=0, now_ms=200 * HOUR_MS)
    make_loader(exchange, tmp_path, since=100 * HOUR_MS).load()

    df = make_loader(exchange, tmp_path, since=50 * HOUR_MS).load()

    assert df["timestamp"].iloc[0] == 50 * HOUR_MS
    assert len(df) == 150
    assert len(pd.read_csv(tmp_path / "binance_BTC-USDT_1h.csv")) == 150


def test_since_accepts_date_string(tmp_path):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    start_ms = int(start.value // 1_000_000)
    exchange = FakeExchange(first_ms=0, now_ms=start_ms + 10 * HOUR_MS)

    df = make_loader(exchange, tmp_path, since="2024-01-01").load()

    assert df["timestamp"].iloc[0] == start_ms
    assert len(df) == 10


def test_retries_network_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("data_pipeline.loaders.time.sleep", lambda _: None)
    exchange = FakeExchange(first_ms=0, now_ms=10 * HOUR_MS, failures=2)

    df = make_loader(exchange, tmp_path).load()

    assert len(df) == 10


def test_gives_up_after_max_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("data_pipeline.loaders.time.sleep", lambda _: None)
    exchange = FakeExchange(first_ms=0, now_ms=10 * HOUR_MS, failures=5)

    with pytest.raises(ccxt.NetworkError):
        make_loader(exchange, tmp_path, max_retries=3).load()


def test_caching_can_be_disabled(tmp_path):
    exchange = FakeExchange(first_ms=0, now_ms=10 * HOUR_MS)
    loader = CCXTLoader("binance", "BTC/USDT", "1h", exchange=exchange, cache_dir=None)

    assert len(loader.load()) == 10
    assert loader.cache_path is None


def test_unknown_exchange_rejected():
    with pytest.raises(ValueError):
        CCXTLoader("not-an-exchange", "BTC/USDT", "1h")
