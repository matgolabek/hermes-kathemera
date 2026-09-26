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
3. `preprocess.split_chronological` splits by time (oldest data for training), and `preprocess.add_target` labels each candle by the log return over the next `TARGET_HORIZON` candles. Labels are computed within each split.
   - `LABEL_MODE=three_class` (default): up/flat/down; moves within ±`FLAT_THRESHOLD` count as flat.
   - `LABEL_MODE=binary`: down (a drop below −`FLAT_THRESHOLD`) vs rest. This matches long-only trading, where the only question is whether to step aside. Keep the threshold above the round-trip cost (0.3% by default), or avoided drops will not pay for the exit.
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

## Training

```bash
python -m training.run                    # train and compare every model
python -m training.run --models lstm gru  # a subset
python -m training.run --eval-test        # also score the held-out test split
```

Every model gets the same data, seed and settings: AdamW (`LEARNING_RATE`, `WEIGHT_DECAY`), gradient clipping (`GRAD_CLIP`), and early stopping once validation loss has not improved for `PATIENCE` epochs. The best epoch is saved to `CHECKPOINT_DIR/<model>.pt` and all metrics to `CHECKPOINT_DIR/results.json`.

The comparison table reports, on the validation split:

- `log_loss`: cross-entropy, the quantity being optimized (lower is better).
- `skill`: improvement in log loss over always predicting the training class frequencies. **At or below 0 means the model learned nothing usable**, which is the expected result for most setups on real price data.
- `acc` / `bal_acc`: accuracy and balanced accuracy (mean per-class recall), plus how often each class is predicted. Compare accuracy with the printed majority-class accuracy, not with 33%.
- `dir_ic`: rank correlation between how bullish the prediction is (P(up) − P(down), or −P(down) for binary labels) and the actual forward return. This is the directional information; values below the printed noise level are luck.
- `vol_ic`: rank correlation between how strongly a move is expected (1 − P(flat), or P(down) for binary labels) and the size of the actual move. High `vol_ic` with `dir_ic` near 0 means the model knows *when* the price moves but not *which way*.
- `dir_acc`: among up/down predictions, how often the return had that sign.

Keep the test split for the final decision: every look at it makes it a less honest estimate.

## Backtest

```bash
python -m backtest.run                      # every trained model on the validation split
python -m backtest.run --models lstm gru
python -m backtest.run --allow-short --min-confidence 0.5
python -m backtest.run --split test         # final check only
```

Each model's checkpoint is loaded with its own preprocessing (features, scaler, horizon, threshold), and its predictions are traded candle by candle:

- `STRATEGY_RULE=argmax` (default): the most likely class sets the position: up (or rest) → long, down → short (with `ALLOW_SHORT`) or out, flat → out. Predictions below `MIN_CONFIDENCE` stay out.
- `STRATEGY_RULE=exit-on-down` (`--rule exit-on-down`): long by default, out (or short) whenever P(down) > `DOWN_PROB_THRESHOLD` (`--down-threshold`). Works with both label modes; pick the threshold on the validation split only.
- A position is taken at a candle's close and held until the next close.
- Every position change costs `FEE_RATE + SLIPPAGE` (default 0.1% + 0.05%) of the traded value; a long-to-short flip counts twice, and any open position is closed at the end.
- Shorts ignore borrow and funding costs, so short results are optimistic.

The table compares each model with buy-and-hold on the same candles: total and annualized return, Sharpe ratio, maximum drawdown, share of time in the market, number of trades, hit rate and total costs, plus the same return and Sharpe before costs. A model that is positive before costs and negative after has a signal too small to trade at that frequency. Equity curves and metrics are saved as `CHECKPOINT_DIR/backtest_<split>_equity.csv` and `backtest_<split>.json`.

The backtest refuses to run if the chosen split overlaps the model's training period, which happens if the data (`SINCE`, the cache) changed after training; retrain in that case.

## Walk-forward evaluation

```bash
python -m backtest.walk_forward                              # all models, 5 folds
python -m backtest.walk_forward --models linear gru --folds 4
python -m backtest.walk_forward --rule exit-on-down --down-threshold 0.4
```

One validation period can be lucky. Walk-forward keeps the first half of the data (`--min-train`) for training only and cuts the rest into `--folds` consecutive periods. For each period, every model is retrained from scratch on all earlier data (its last 10% for early stopping) and then scored and traded on that period, so every result is out-of-sample. The report shows each fold, and all folds stitched into one out-of-sample equity curve per model, with how many folds beat buy-and-hold. Output goes to `CHECKPOINT_DIR/walk_forward/`.

Training runs once per model and fold, so this takes `folds` times longer than `python -m training.run`.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Run

```bash
python main.py
```
