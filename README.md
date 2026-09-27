# hermes-kathemera

An experiment in trading Bitcoin with machine learning, built to answer one question honestly: **can LSTM/GRU models trained on hourly BTC/USDT candles beat simply buying and holding, after trading costs?**

**Short answer: no.** Across six experiments, two label schemes and five walk-forward periods (May 2023 – September 2026), no model-driven strategy beat buy-and-hold after costs. The models reliably learned *when* the price is about to move (volatility), but not *which way*, and trading on that signal cost more than it saved.

The repository is complete as a research pipeline (data download, features, models, training, cost-aware backtests, walk-forward evaluation) and is published with its results, including the negative ones.

> **Disclaimer.** This is a research and learning project, not financial advice and not a trading system. The live-trading package (`execution/`) is intentionally left as an unimplemented stub. Do not trade real money based on this code.

## Contents

- [Results](#results)
- [What we learned](#what-we-learned)
- [Limitations](#limitations)
- [Reproducing the results](#reproducing-the-results)
- [Project structure](#project-structure)
- [Setup](#setup)
- [How it works](#how-it-works)
- [Tests](#tests)
- [References](#references)
- [License](#license)

## Results

All numbers come from the JSON files in [`results/`](results/), produced by the commands in [Reproducing the results](#reproducing-the-results).

### Setup of every experiment

- **Data**: Binance BTC/USDT, 1-hour candles, January 2020 – September 2026 (~59,000 candles), downloaded with CCXT.
- **Inputs**: windows of the last 64 candles × 16 scale-free features (returns, candle shape, RSI, MACD, ATR, moving-average distance, Bollinger %B, volatility, relative volume, hour and weekday).
- **Models**: `constant` (class frequencies only), `linear`, `mlp`, `lstm`, `gru`, all trained with the same settings, seed and early stopping.
- **Costs**: 0.1% fee + 0.05% slippage per trade, so 0.3% per round trip (buy and sell).
- **Trading**: long-only spot; a position is taken at a candle's close and held for one candle.
- **Benchmark**: buy-and-hold over the same candles.

### Overview

| # | Question | Labels | Strategy | Verdict |
|---|---|---|---|---|
| 1 | Can the models predict the next hour's direction? | up / flat / down (±0.2%) | long when "up" is most likely | Loses 36–59% after costs |
| 2 | What exactly do the models learn? | down (< −0.5%) vs rest | direction and volatility metrics | Volatility yes, direction no |
| 3 | Does stepping aside before predicted drops help? | down vs rest | exit if P(down) > 0.4 | Rule almost never fires; ≈ buy-and-hold |
| 4 | Same, with an exit threshold that adapts to each model | down vs rest | exit in P(down)'s top 20% | Helps before costs, −91% after |
| 5 | Same, with hysteresis to cut trading | down vs rest | exit top 20%, re-enter below top 50% | Still −70%; a simple volatility rule does better |
| 6 | Same, on a 24-hour horizon | down (< −3% in 24h) vs rest | as in 5 | Best model +3% vs +215% for buy-and-hold |

### 1. Predicting direction (fixed train/validation/test split)

Training on 2020 – mid-2025, validating on ≈ May 2025 – January 2026 and testing on ≈ January – September 2026. The models beat the no-information baseline on log loss (`skill` 7.6–8.8%), but mostly by predicting "flat" (87–91% of the time), with accuracy barely above always guessing flat (56% vs 55%).

| Validation | Return | Sharpe | Max drawdown | Trades | Return before costs |
|---|---|---|---|---|---|
| buy & hold | −16.6% | −0.53 | −34.8% | 2 | −16.3% |
| linear | −36.2% | −5.35 | −36.5% | 292 | −1.1% |
| mlp | −52.7% | −5.87 | −53.8% | 428 | −10.1% |
| lstm | −46.9% | −5.31 | −47.3% | 342 | −11.3% |
| gru | −43.1% | −4.61 | −43.6% | 378 | +0.3% |

On the test period, buy-and-hold returned −4.5%, while the models returned −52% to −59% after costs. Before costs they ranged from −13.5% to +20.7%, and the sign flipped between validation and test for the same model: noise, not signal. The average profit per round trip before costs was between −0.07% and +0.06%, against a cost of 0.30%. The best strategy was the `constant` model, which never traded.

### 2. Direction or volatility?

Binary labels ("does the price drop more than 0.5% in the next hour?", 7% of hours) and two diagnostics: **direction IC**, the rank correlation between the model's bullishness and the next return, and **volatility IC**, the rank correlation between its expected move size and the actual move size. Correlations below 0.026 are indistinguishable from chance at this sample size.

| Model | Skill | Direction IC | Volatility IC |
|---|---|---|---|
| linear | +12.4% | +0.013 | **0.31** |
| mlp | +15.7% | +0.003 | **0.37** |
| lstm | +16.3% | +0.014 | **0.39** |
| gru | +16.1% | +0.010 | **0.39** |

The skill came entirely from volatility: a large drop in the next hour is mostly a large move in the next hour. This pattern held in every later experiment and every walk-forward period: volatility IC between 0.10 and 0.40, direction IC close to zero with signs that flipped from period to period.

### 3–6. Exit filters, walk-forward

From here on, every result is **walk-forward**: the first half of the data is used only for training, the rest is cut into five consecutive ~8-month periods, and for each period every model is retrained from scratch on all earlier data and traded on that period. The five periods are joined into one out-of-sample record (May 2023 – September 2026). `vol_rule` is a non-ML baseline: the same exit rule, driven by recent realized volatility instead of a model.

| Experiment | Strategy | Return | Sharpe | Max drawdown | Trades | Time in market | Periods beating B&H |
|---|---|---|---|---|---|---|---|
| — | **buy & hold** | **+230%** | **1.01** | −53% | 10 | 100% | — |
| 3: exit if P(down) > 0.4 | gru | +212% | 0.98 | −56% | 22 | 100% | 1/5 |
| | linear | +188% | 0.93 | −55% | 64 | 100% | 0/5 |
| 4: exit in top 20% | gru | −91% | −2.08 | −94% | 2,350 | 80% | 0/5 |
| | linear | −99% | −4.14 | −99% | 3,842 | 80% | 0/5 |
| 5: + hysteresis | vol_rule | +21% | 0.34 | −51% | 280 | 70% | 0/5 |
| | gru | −70% | −1.17 | −78% | 1,186 | 66% | 0/5 |
| | linear | −91% | −2.21 | −92% | 2,069 | 69% | 0/5 |
| 6: + 24h horizon | vol_rule | +14% | 0.29 | −53% | 274 | 70% | 0/5 |
| | gru | +3% | 0.18 | −45% | 438 | 66% | 1/5 |
| | linear | −50% | −0.46 | −70% | 1,094 | 68% | 0/5 |

(Buy-and-hold made +215% over the slightly shorter period of experiment 6.)

In experiment 3, a fixed probability threshold almost never triggered, because "down" is a rare class, so the strategies were buy-and-hold with a few costly extra trades.

Experiment 4 is the most informative one. **Before costs**, the GRU exit filter behaved like a real risk filter:

| Period | B&H return | B&H Sharpe | B&H max DD | GRU return before costs | GRU Sharpe before costs | GRU max DD before costs | GRU return after costs |
|---|---|---|---|---|---|---|---|
| 2023-05 – 2024-01 | +59% | 1.98 | −21% | +58% | **2.27** | **−13%** | +15% |
| 2024-01 – 2024-09 | +49% | 1.38 | −32% | +30% | 1.39 | **−17%** | −48% |
| 2024-09 – 2025-05 | +76% | 1.91 | −31% | +43% | 1.66 | **−28%** | −35% |
| 2025-05 – 2026-01 | −17% | −0.55 | −35% | −3% | **0.01** | **−23%** | −39% |
| 2026-01 – 2026-09 | −5% | 0.05 | −35% | +5% | **0.40** | **−22%** | −64% |

Lower drawdowns in all five periods and a higher Sharpe ratio in four, before costs. But the position flipped in and out around 700 times a year, and at 0.3% per round trip the costs consumed that edge many times over. Hysteresis (experiment 5) and a 24-hour horizon (experiment 6) cut trading by up to five times, yet not enough, and time spent out of a strongly rising market cost more than the avoided volatility saved.

## What we learned

1. **Hourly BTC direction was not predictable with these features.** Direction IC stayed within noise in every setup; this matches the literature on market efficiency.
2. **Volatility is predictable, and ML models learn it well**, with volatility IC up to 0.40. It is a real, repeatable signal, but not a profitable one on its own for a long-only in/out strategy.
3. **Trading costs decide everything at hourly frequency.** Several strategies were profitable or better than buy-and-hold *before* costs and deeply negative after. An edge has to exceed 0.3% per round trip.
4. **Volatility filters do not protect against the drawdowns that matter.** The large 2025–2026 decline was a slow, months-long fall. A volatility filter leaves during sudden turbulence (often upward in crypto) and stays in during a calm decline: the stitched maximum drawdown of `vol_rule` (−51% to −53%) was the same as buy-and-hold's (−53%).
5. **Simple baselines are essential.** The `constant` model (never trading) beat every model in experiment 1, and the non-ML `vol_rule` beat both neural networks in experiment 5. Without these comparisons, the neural networks' "skill" of up to 16% would have looked like success.
6. **Evaluation discipline matters more than model choice.** The same MLP lost 10% before costs on the validation period and gained 21% (Sharpe 1.31) on the test period. Any single period can mislead; walk-forward over several periods is what made the conclusions trustworthy.

## Limitations

- **One market, one exchange, one frequency.** Only BTC/USDT on Binance with hourly candles was tested; daily data, other assets, order-book or on-chain data might behave differently.
- **Only one training run per model and period** (fixed seed). Differences between models of a few hundredths of Sharpe are within the noise of training.
- **Costs are a simple model**: a fixed 0.15% per trade, no market impact, and trades at the candle close. Lower fees (e.g. maker orders) would reduce but not remove the cost problem at the observed trade counts.
- **The test split of experiment 1 was looked at**, so it is no longer untouched; the last walk-forward period overlaps it.
- **The IC noise level stored in the results of experiment 6 is understated.** The files report 0.026, but 24-hour returns measured every hour overlap, so the correct level is about 0.13 (fixed in the code since). The direction ICs between −0.08 and +0.15 in that experiment are therefore at or within the noise level, and their signs flip between periods.
- **Not tried**: trend-following on daily data (e.g. holding only above a 100–200-day moving average, the most promising idea left, since it targets slow declines), volatility targeting with fractional positions, and shorting with realistic funding costs.

## Reproducing the results

Numbers will differ slightly because the downloaded history grows every hour. Commands use the defaults in `.env.example` unless shown otherwise.

```bash
# 1. Three-class labels, fixed split
python -m training.run
python -m backtest.run                   # validation split
python -m backtest.run --split test      # test split, once

# 2. Binary labels, direction/volatility diagnostics
LABEL_MODE=binary FLAT_THRESHOLD=0.005 python -m training.run
LABEL_MODE=binary FLAT_THRESHOLD=0.005 python -m backtest.run --rule exit-on-down --down-threshold 0.5

# 3-5. Walk-forward, binary labels, 1-hour horizon
export LABEL_MODE=binary FLAT_THRESHOLD=0.005
python -m backtest.walk_forward --models linear gru --rule exit-on-down --down-threshold 0.4
python -m backtest.walk_forward --models linear gru --exit-share 0.2
python -m backtest.walk_forward --models linear gru --exit-share 0.2 --reentry-share 0.5

# 6. Walk-forward, 24-hour horizon
LABEL_MODE=binary FLAT_THRESHOLD=0.03 TARGET_HORIZON=24 \
  python -m backtest.walk_forward --models linear gru --exit-share 0.2 --reentry-share 0.5
```

The first download of hourly data since 2020 takes around 60 requests. Walk-forward retrains each model five times, so it takes a while on a CPU.

## Project structure

```
data_pipeline/   download (CCXT, cached), cleaning, indicators, labels, splits, walk-forward folds
models/          constant, linear and MLP baselines; LSTM/GRU recurrent model; factory
training/        training loop, metrics (skill, direction/volatility IC), checkpoints, runner
backtest/        cost-aware simulation, strategy rules, backtest and walk-forward runners
execution/       live-trading interfaces: intentionally unimplemented stubs
results/         JSON results of the experiments above
tests/           unit and end-to-end tests (no network needed)
config.py        settings read from environment variables (see .env.example)
main.py          downloads data, prints dataset shapes and builds the configured model
```

## Setup

Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All settings are environment variables with defaults; `.env.example` lists and explains them. Set the ones you want to change in your shell (or with a tool that loads `.env` files). No API key is needed: market data is public, and nothing here places orders.

```bash
python main.py   # downloads data, prints dataset shapes and class counts, builds the model
```

## How it works

### Market data

`data_pipeline.loaders.CCXTLoader` downloads OHLCV candles through [CCXT](https://github.com/ccxt/ccxt), a library that talks to many exchanges' public APIs. It pages through history, since exchanges return a limited number of candles per request, and caches the result as CSV in `data/cache/`. Later runs only download candles newer than the cache. Only closed candles are returned; timestamps are candle open times in UTC milliseconds. Some exchanges (e.g. Kraken) only serve recent candles through this API, so deep history is best fetched from exchanges like Binance.

```python
from config import get_data_config
from data_pipeline.loaders import CCXTLoader

cfg = get_data_config()
df = CCXTLoader(cfg.exchange, cfg.symbol, cfg.timeframe, since=cfg.since, cache_dir=cfg.cache_dir).load()
```

### From candles to model inputs

`data_pipeline.pipeline.prepare_datasets` turns raw candles into train/validation/test arrays:

1. `preprocess.clean_ohlcv` parses timestamps into a UTC index, sorts, removes duplicates and invalid candles, and fills missing candles with the previous close and zero volume (flagged in `is_filled`).
2. `indicators.add_technical_indicators` adds the 16 `FEATURE_COLUMNS`. All are scale-free and use only past candles.
3. `preprocess.split_chronological` splits by time (oldest data for training), and `preprocess.add_target` labels each candle by the log return over the next `TARGET_HORIZON` candles. Labels are computed within each split, so no label looks into the next split.
   - `LABEL_MODE=three_class` (default): up/flat/down; moves within ±`FLAT_THRESHOLD` count as flat.
   - `LABEL_MODE=binary`: down (a drop below −`FLAT_THRESHOLD`) vs rest, matching long-only trading, where the only question is whether to step aside. Keep the threshold above the round-trip cost (0.3% by default).
4. `preprocess.FeatureScaler` standardizes features using training-period statistics only.
5. `preprocess.make_supervised_sequences` builds sliding windows: `X` has shape `(samples, SEQUENCE_LENGTH, 16)`, `y` holds class indices for `nn.CrossEntropyLoss`, and the forward returns are kept for evaluating trades.

### Models

Set `MODEL_NAME` to pick a model. All map `(batch, SEQUENCE_LENGTH, features)` to one logit per class (3 for `three_class`, 2 for `binary`).

| `MODEL_NAME` | Model |
|---|---|
| `constant` | Baseline: ignores the input and learns class frequencies |
| `linear` | Baseline: logistic regression over the flattened window |
| `mlp` | Baseline: one hidden layer over the flattened window |
| `lstm`, `gru` | Recurrent network; last hidden state through LayerNorm, dropout and a linear head |

Checkpoints store the model config and the preprocessing settings (features, fitted scaler, horizon, threshold, label mode) next to the weights, so `training.checkpoint.load_model(path)` rebuilds a trained model without any other files.

### Training

```bash
python -m training.run                    # train and compare every model
python -m training.run --models lstm gru  # a subset
python -m training.run --eval-test        # also score the held-out test split
```

Every model gets the same data, seed and settings: AdamW (`LEARNING_RATE`, `WEIGHT_DECAY`), gradient clipping (`GRAD_CLIP`), and early stopping once validation loss has not improved for `PATIENCE` epochs. The best epoch is saved to `CHECKPOINT_DIR/<model>.pt` and all metrics to `CHECKPOINT_DIR/results.json`. The comparison table reports:

- `log_loss` and `skill`: improvement in log loss over always predicting the training class frequencies. At or below 0 means the model learned nothing usable.
- `acc` / `bal_acc`: accuracy and balanced accuracy, plus how often each class is predicted. Compare accuracy with the printed majority-class accuracy.
- `dir_ic`: rank correlation between how bullish the prediction is (P(up) − P(down), or −P(down) for binary labels) and the actual forward return.
- `vol_ic`: rank correlation between how strongly a move is expected (1 − P(flat), or P(down) for binary labels) and the size of the actual move. High `vol_ic` with `dir_ic` near 0 means the model knows *when* the price moves but not *which way*.
- The printed noise level, 2/√(samples / horizon): correlations below it are likely luck.
- `dir_acc`: among up/down predictions, how often the return had that sign.

Keep the test split for the final decision: every look at it makes it a less honest estimate.

### Backtest

```bash
python -m backtest.run                      # every trained model on the validation split
python -m backtest.run --models lstm gru
python -m backtest.run --allow-short --min-confidence 0.5
python -m backtest.run --exit-share 0.2 --reentry-share 0.5
python -m backtest.run --split test         # final check only
```

Each model's checkpoint is loaded with its own preprocessing, and its predictions are traded candle by candle:

- `STRATEGY_RULE=argmax` (default): the most likely class sets the position: up (or rest) → long, down → short (with `ALLOW_SHORT`) or out, flat → out. Predictions below `MIN_CONFIDENCE` stay out.
- `STRATEGY_RULE=exit-on-down` (`--rule exit-on-down`): long by default, out (or short) whenever P(down) > `DOWN_PROB_THRESHOLD` (`--down-threshold`).
- `EXIT_SHARE` (`--exit-share 0.2`): exit-on-down with a threshold set so that the top 20% of P(down) values lead to an exit. The threshold is calibrated on the model's predictions for the data just before the traded period (the last training windows for `val`, the validation split for `test`, each period's validation block in walk-forward), never on the traded period itself.
- `REENTRY_SHARE` (`--reentry-share 0.5`, with `EXIT_SHARE`): hysteresis. After an exit, stay out until P(down) has left its top 50%, instead of re-entering the moment it dips below the exit level.
- With `EXIT_SHARE`, results also include `vol_rule`: the same rule driven by realized volatility (`volatility_24`) instead of a model.
- A position is taken at a candle's close and held until the next close. Every position change costs `FEE_RATE + SLIPPAGE` (default 0.1% + 0.05%) of the traded value; a long-to-short flip counts twice, and any open position is closed at the end. Shorts ignore borrow and funding costs.

The table compares each strategy with buy-and-hold on the same candles: return, annualized return, Sharpe ratio, maximum drawdown, time in the market, trades, hit rate and total costs, plus return and Sharpe before costs. Results are saved per strategy setting, e.g. `CHECKPOINT_DIR/backtest_val_exit_top20pct_re50pct.json` and `..._equity.csv`. The backtest refuses to run if the chosen split overlaps the model's training period, which happens if the data changed after training.

### Walk-forward evaluation

```bash
python -m backtest.walk_forward                              # all models, 5 periods
python -m backtest.walk_forward --models linear gru --folds 4
python -m backtest.walk_forward --exit-share 0.2 --reentry-share 0.5
```

The first half of the data (`--min-train`) is only used for training; the rest is cut into `--folds` consecutive periods. For each period, every model is retrained from scratch on all earlier data (its last 10% for early stopping) and then scored and traded on that period, so every result is out-of-sample. The report shows each period, and all periods joined into one out-of-sample equity curve per strategy, with how many periods beat buy-and-hold. Output goes to `CHECKPOINT_DIR/walk_forward/<label mode>_h<horizon>_<strategy>/`.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests use simulated exchanges and synthetic prices, so they need no network access. They also run on every push via GitHub Actions.

## References

- Fama, E. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. *Notices of the AMS*.
- Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
- Harvey, C., Liu, Y. & Zhu, H. (2016). …and the Cross-Section of Expected Returns. *Review of Financial Studies*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Fischer, T. & Krauss, C. (2018). Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions. *European Journal of Operational Research*.
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*.
- Liu, Y. & Tsyvinski, A. (2021). Risks and Returns of Cryptocurrency. *Review of Financial Studies*.
- Moreira, A. & Muir, T. (2017). Volatility-Managed Portfolios. *Journal of Finance*.
- Zeng, A., Chen, M., Zhang, L. & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI*.

## License

[MIT](LICENSE)
