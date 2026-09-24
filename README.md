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

## Run

```bash
python main.py
```
