"""Technical indicator feature engineering helpers.

Every feature only uses the current and earlier candles, so nothing leaks from the
future. Features are scale-free (returns, ratios, bounded oscillators) rather than raw
prices, so a model trained on BTC at $10k still applies at $100k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "hl_range",
    "body",
    "rsi_14",
    "macd",
    "macd_hist",
    "natr_14",
    "close_sma_20",
    "close_sma_50",
    "bb_pctb_20",
    "volatility_24",
    "volume_z_24",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

_FLAT_TOL = 1e-6


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``FEATURE_COLUMNS`` to a cleaned OHLCV frame with a UTC ``DatetimeIndex``.

    Early rows lack enough history for the longer windows and hold NaN; drop them with
    ``df.dropna(subset=FEATURE_COLUMNS)`` before building sequences.
    """
    df = df.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    log_close = np.log(close)

    # Candle shape
    df["log_return"] = log_close.diff()
    df["hl_range"] = (high - low) / close
    df["body"] = (close - df["open"]) / df["open"]

    # Momentum
    df["rsi_14"] = _rsi(close, 14) / 100.0
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd / close
    df["macd_hist"] = (macd - signal) / close

    # Volatility
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["natr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close
    df["volatility_24"] = df["log_return"].rolling(24).std()

    # Trend: distance from moving averages
    df["close_sma_20"] = close / close.rolling(20).mean() - 1
    df["close_sma_50"] = close / close.rolling(50).mean() - 1
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    # Flat price or volume (e.g. filled gaps) gives zero spread; use the neutral value.
    # Rolling std of a flat window is float noise rather than exactly 0, hence the tolerance.
    df["bb_pctb_20"] = ((close - (sma_20 - 2 * std_20)) / (4 * std_20)).where(~(std_20 <= close * _FLAT_TOL), 0.5)

    # Volume relative to its recent level (log scale tames spikes)
    log_volume = np.log1p(volume)
    volume_std = log_volume.rolling(24).std()
    volume_z = (log_volume - log_volume.rolling(24).mean()) / volume_std
    df["volume_z_24"] = volume_z.where(~(volume_std <= _FLAT_TOL), 0.0)

    # Calendar: crypto trades 24/7 but activity follows time of day and weekday
    hours = df.index.hour + df.index.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    return df


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's relative strength index (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    # No losses in the window: RSI is 100 (or 50 when price did not move at all).
    rsi = rsi.where(loss != 0, np.where(gain == 0, 50.0, 100.0))
    return rsi.where(gain.notna())
