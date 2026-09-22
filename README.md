# hermes-kathemera

Initial modular scaffold for a time-series ML trading system with separated concerns:

- `data_pipeline/` for OHLCV loading, cleaning, and indicator features
- `models/` for swappable LSTM/GRU PyTorch models
- `training/` for datasets, loops, and checkpoints
- `backtest/` for out-of-sample evaluation and financial metrics
- `execution/` for exchange integration and risk-managed order execution

## Configuration

Copy `.env.example` values into your environment before running code.

## Run

```bash
python main.py
```
