"""Walk-forward evaluation: retrain and trade over several consecutive periods.

Usage::

    python -m backtest.walk_forward                          # all models, 5 folds
    python -m backtest.walk_forward --models linear gru --folds 4
    python -m backtest.walk_forward --rule exit-on-down --down-threshold 0.4

The first ``--min-train`` share of the data is only used for training. The rest is
cut into ``--folds`` consecutive periods. For each period, every model is trained
from scratch on all data before it (the last ``--val-fraction`` of that history is
used for early stopping) and then scored and traded on the period. A single
validation period can be lucky; a result that holds in most folds is more
believable. The per-fold returns are also stitched into one out-of-sample equity
curve per model.

Results go to ``CHECKPOINT_DIR/walk_forward/``: fold checkpoints, ``walk_forward.json``
and ``equity.csv``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from backtest.evaluator import BacktestResult
from backtest.metrics import calculate_metrics, periods_per_year
from backtest.run import BUY_AND_HOLD, add_strategy_args, backtest_split, buy_and_hold, describe_rule, strategy_config
from config import BacktestConfig, ModelConfig, TrainingConfig, get_data_config, get_model_config, get_training_config
from data_pipeline.loaders import CCXTLoader
from data_pipeline.pipeline import PreparedData, prepare_walk_forward
from models import MODEL_NAMES
from training.checkpoint import load_model
from training.run import train_and_evaluate

logger = logging.getLogger(__name__)


def run_fold(
    fold: int,
    data: PreparedData,
    models: Sequence[str],
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    backtest_cfg: BacktestConfig,
    periods: float,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, BacktestResult]]:
    """Train every model on one fold's history and trade its evaluation period."""
    fold_train_cfg = dataclasses.replace(train_cfg, checkpoint_dir=out_dir / f"fold_{fold}")
    summary: dict[str, Any] = {
        "fold": fold,
        "train_period": [str(data.train.timestamps[0]), str(data.train.timestamps[-1])],
        "eval_period": [str(data.test.timestamps[0]), str(data.test.timestamps[-1])],
        "models": {},
    }
    backtests = {BUY_AND_HOLD: buy_and_hold(data.test, backtest_cfg, periods)}
    summary["models"][BUY_AND_HOLD] = {"backtest": dataclasses.asdict(backtests[BUY_AND_HOLD].metrics)}

    for name in models:
        logger.info("=== Fold %d: training %s", fold, name)
        result = train_and_evaluate(data, dataclasses.replace(model_cfg, model_name=name), fold_train_cfg, eval_test=True)
        model, _ = load_model(Path(result["checkpoint"]), train_cfg.device)
        backtests[name] = backtest_split(model, data.test, data.class_names, backtest_cfg, periods, train_cfg.device)
        summary["models"][name] = {
            "best_epoch": result["best_epoch"],
            "classification": result["test"],
            "backtest": dataclasses.asdict(backtests[name].metrics),
            "gross": dataclasses.asdict(backtests[name].gross_metrics),
        }
    return summary, backtests


def stitch(results: Sequence[BacktestResult], periods: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join consecutive fold backtests into one out-of-sample record and summarize it.

    Each fold closes its position at the end and reopens in the next, so the stitched
    record pays a little more in costs than one continuous run would. The first
    ``sequence_length - 1`` candles of every fold only serve as window context, so
    there is a short gap between consecutive folds.
    """
    frame = pd.concat([r.frame for r in results])
    net = frame["net_return"].to_numpy()
    frame = frame.assign(equity=np.cumprod(1.0 + net))
    trades = int(np.count_nonzero(frame["turnover"]))
    metrics = calculate_metrics(net, frame["position"].to_numpy(), frame["cost"].to_numpy(), trades, periods)
    return frame, dataclasses.asdict(metrics)


def aggregate(folds: Sequence[dict[str, Any]], strategies: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per-strategy averages across folds and how often it beat buy-and-hold."""
    summary = {}
    for name in strategies:
        per_fold = [f["models"][name] for f in folds]
        sharpes = [m["backtest"]["sharpe"] for m in per_fold]
        bh_sharpes = [f["models"][BUY_AND_HOLD]["backtest"]["sharpe"] for f in folds]
        entry: dict[str, Any] = {
            "mean_sharpe": float(np.mean(sharpes)),
            "folds_beating_buy_and_hold": int(sum(s > b for s, b in zip(sharpes, bh_sharpes))),
        }
        if "classification" in per_fold[0]:
            for key in ("skill", "direction_ic", "volatility_ic"):
                entry[f"mean_{key}"] = float(np.mean([m["classification"][key] for m in per_fold]))
        summary[name] = entry
    return summary


def format_report(folds: Sequence[dict[str, Any]], totals: dict[str, dict[str, Any]], backtest_cfg: BacktestConfig) -> str:
    """Render per-fold results and the stitched out-of-sample summary."""
    lines = [f"Walk-forward, after costs of {backtest_cfg.cost_per_trade:.2%} per trade ({describe_rule(backtest_cfg)})", ""]
    header = f"{'fold':>4s}  {'evaluation period':23s} {'strategy':10s} {'skill':>7s} {'dir_ic':>7s} {'vol_ic':>7s} {'return':>8s} {'sharpe':>7s} {'max_dd':>7s} {'trades':>6s}"
    lines += [header, "-" * len(header)]
    for f in folds:
        period = f"{f['eval_period'][0][:10]} - {f['eval_period'][1][:10]}"
        for name, m in f["models"].items():
            c, b = m.get("classification"), m["backtest"]
            cls = f"{c['skill']:>+7.2%} {c['direction_ic']:>+7.3f} {c['volatility_ic']:>+7.3f}" if c else f"{'':>7s} {'':>7s} {'':>7s}"
            lines.append(
                f"{f['fold']:>4d}  {period:23s} {name:10s} {cls} {b['total_return']:>+8.1%} {b['sharpe']:>7.2f} "
                f"{b['max_drawdown']:>7.1%} {b['trades']:>6d}"
            )
            period = ""
        lines.append("")

    header = f"{'strategy':10s} {'return':>8s} {'annual':>8s} {'sharpe':>7s} {'max_dd':>7s} {'exposure':>8s} {'trades':>6s} {'beat b&h':>8s} {'mean dir_ic':>11s}"
    lines += ["All folds stitched together (out-of-sample only):", header, "-" * len(header)]
    n_folds = len(folds)
    for name, t in totals.items():
        s = t["stitched"]
        beat = f"{t['folds_beating_buy_and_hold']}/{n_folds}" if name != BUY_AND_HOLD else "-"
        dir_ic = f"{t['mean_direction_ic']:+.3f}" if "mean_direction_ic" in t else "-"
        lines.append(
            f"{name:10s} {s['total_return']:>+8.1%} {s['annual_return']:>+8.1%} {s['sharpe']:>7.2f} {s['max_drawdown']:>7.1%} "
            f"{s['exposure']:>8.0%} {s['trades']:>6d} {beat:>8s} {dir_ic:>11s}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES, help="models to evaluate")
    parser.add_argument("--folds", type=int, default=5, help="number of evaluation periods")
    parser.add_argument("--min-train", type=float, default=0.5, help="share of data used only for training")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="share of each fold's history for early stopping")
    add_strategy_args(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data_cfg, model_cfg, train_cfg = get_data_config(), get_model_config(), get_training_config()
    backtest_cfg = strategy_config(args)
    periods = periods_per_year(data_cfg.timeframe)
    out_dir = train_cfg.checkpoint_dir / "walk_forward"

    raw = CCXTLoader(
        data_cfg.exchange, data_cfg.symbol, data_cfg.timeframe, since=data_cfg.since, cache_dir=data_cfg.cache_dir
    ).load()
    folds_data = prepare_walk_forward(
        raw,
        n_folds=args.folds,
        sequence_length=data_cfg.sequence_length,
        min_train_fraction=args.min_train,
        val_fraction=args.val_fraction,
        timeframe=data_cfg.timeframe,
        horizon=data_cfg.horizon,
        flat_threshold=data_cfg.flat_threshold,
        label_mode=data_cfg.label_mode,
    )
    model_cfg = dataclasses.replace(
        model_cfg, input_size=len(folds_data[0].feature_columns), output_size=len(folds_data[0].class_names)
    )

    fold_summaries = []
    fold_backtests: dict[str, list[BacktestResult]] = {}
    for k, data in enumerate(folds_data, start=1):
        if min(len(data.train), len(data.val), len(data.test)) == 0:
            raise ValueError(f"Fold {k} has an empty split; use fewer folds or more data")
        summary, backtests = run_fold(k, data, args.models, model_cfg, train_cfg, backtest_cfg, periods, out_dir)
        fold_summaries.append(summary)
        for name, result in backtests.items():
            fold_backtests.setdefault(name, []).append(result)

    totals = aggregate(fold_summaries, list(fold_backtests))
    equity = {}
    for name, results in fold_backtests.items():
        frame, totals[name]["stitched"] = stitch(results, periods)
        equity[name] = frame["equity"]

    print()
    print(format_report(fold_summaries, totals, backtest_cfg))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(equity).to_csv(out_dir / "equity.csv")
    report = {
        "config": {"folds": args.folds, "min_train": args.min_train, "val_fraction": args.val_fraction, **dataclasses.asdict(backtest_cfg)},
        "data": {"label_mode": data_cfg.label_mode, "horizon": data_cfg.horizon, "flat_threshold": data_cfg.flat_threshold},
        "folds": fold_summaries,
        "totals": totals,
    }
    (out_dir / "walk_forward.json").write_text(json.dumps(report, indent=2))
    print(f"\nFold checkpoints, walk_forward.json and equity.csv saved in {out_dir}/")
    return report


if __name__ == "__main__":
    main()
