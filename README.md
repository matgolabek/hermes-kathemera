# hermes-kathemera

Initial modular scaffold for a time-series ML trading system with separated concerns:

- `data_pipeline/` for OHLCV loading, cleaning, and indicator features
- `models/` for swappable LSTM/GRU PyTorch models
- `training/` for datasets, loops, and checkpoints
- `backtest/` for out-of-sample evaluation and financial metrics
- `execution/` for exchange integration and risk-managed order execution

## Setup

Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Deactivate the virtual environment when finished:

```bash
deactivate
```

## Configuration

Copy `.env.example` values into your environment before running code.

## Market data

`data_pipeline.loaders.CCXTLoader` downloads OHLCV candles through [CCXT](https://github.com/ccxt/ccxt), a library that talks to many exchanges' public APIs (no API key is needed for market data). It pages through history, since exchanges return a limited number of candles per request, and caches the result as CSV in `data/cache/`. Later runs only download candles newer than the cache. Only closed candles are returned; timestamps are candle open times in UTC milliseconds.

```python
from config import get_data_config
from data_pipeline.loaders import CCXTLoader

cfg = get_data_config()
df = CCXTLoader(cfg.exchange, cfg.symbol, cfg.timeframe, since=cfg.since, cache_dir=cfg.cache_dir).load()
```

The first download of hourly data since 2020 takes around 60 requests. Some exchanges (e.g. Kraken) only serve recent candles through this API, so deep history is best fetched from exchanges like Binance.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Run

```bash
python main.py
```
