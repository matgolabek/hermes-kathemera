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

### From candles to model inputs

`data_pipeline.pipeline.prepare_datasets` turns raw candles into train/validation/test arrays:

1. `preprocess.clean_ohlcv` parses timestamps into a UTC index, sorts, removes duplicates and invalid candles, and fills missing candles with the previous close and zero volume (flagged in `is_filled`).
2. `indicators.add_technical_indicators` adds the 16 `FEATURE_COLUMNS`: returns, candle shape, RSI, MACD, ATR, moving-average distance, Bollinger %B, volatility, relative volume and time-of-day/weekday. All are scale-free and use only past candles.
3. `preprocess.split_chronological` splits by time (oldest data for training), and `preprocess.add_target` labels each candle up/flat/down by the log return over the next `TARGET_HORIZON` candles; moves within ±`FLAT_THRESHOLD` (default 0.2%, about a round-trip fee) count as flat. Labels are computed within each split.
4. `preprocess.FeatureScaler` standardizes features using training-period statistics only.
5. `preprocess.make_supervised_sequences` builds sliding windows: `X` has shape `(samples, SEQUENCE_LENGTH, 16)`, `y` holds class indices `(samples,)` for `nn.CrossEntropyLoss`, and `returns` keeps the underlying forward returns for evaluating trades.

`python main.py` downloads the configured market, prints dataset shapes and class counts, and builds the configured model.

## Models

Set `MODEL_NAME` to pick a model. All map `(batch, SEQUENCE_LENGTH, features)` to 3 class logits (down, flat, up).

| `MODEL_NAME` | Model |
|---|---|
| `constant` | Baseline: ignores the input and learns class frequencies |
| `linear` | Baseline: logistic regression over the flattened window |
| `mlp` | Baseline: one hidden layer over the flattened window |
| `lstm`, `gru` | Recurrent network, last hidden state through LayerNorm, dropout and a linear head |

A recurrent model is only useful if it beats the baselines on validation data.

Checkpoints store, next to the weights, the model config and the preprocessing settings (`PreparedData.metadata()`: feature list, fitted scaler, horizon, threshold), so `training.checkpoint.load_model(path)` rebuilds a trained model without any other files.

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
