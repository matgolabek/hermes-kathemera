"""Backtest trained models against buy-and-hold, after trading costs.

Usage::

    python -m backtest.run                      # every trained model, validation split
    python -m backtest.run --models lstm gru    # a subset
    python -m backtest.run --split test         # the held-out test split (use once, at the end)
    python -m backtest.run --allow-short --min-confidence 0.5
    python -m backtest.run --rule exit-on-down --down-threshold 0.4
    python -m backtest.run --exit-share 0.1     # exit for the top 10% of P(down)

Reads checkpoints written by ``python -m training.run``, rebuilds each model's inputs
with the preprocessing stored in its checkpoint (same features, scaler, horizon and
threshold), and simulates trading its predictions. Results and equity curves are
saved next to the checkpoints.
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
from torch import nn

from backtest.evaluator import (
    STRATEGY_RULES,
    BacktestResult,
    calibrate_down_threshold,
    positions_from_probs,
    run_backtest,
)
from backtest.metrics import periods_per_year
from config import BacktestConfig, get_backtest_config, get_data_config, get_training_config
from data_pipeline.loaders import CCXTLoader
from data_pipeline.pipeline import PreparedData, SequenceSet, prepare_datasets
from data_pipeline.preprocess import CLASS_NAMES, FeatureScaler
from models import MODEL_NAMES
from training.checkpoint import load_model
from training.run import make_loader
from training.train import predict_proba

logger = logging.getLogger(__name__)

BUY_AND_HOLD = "buy&hold"


def check_no_overlap(split: SequenceSet, metadata: dict[str, Any], split_name: str) -> None:
    """Refuse to backtest on candles the model was trained on.

    Splits are fractions of the loaded data, so if more candles were downloaded after
    training, the validation/test periods move and can overlap the training period.
    """
    trained_until = metadata.get("train_period", [None, None])[1]
    if trained_until is not None and len(split) and split.timestamps[0] <= pd.Timestamp(trained_until):
        raise ValueError(
            f"The {split_name} split starts at {split.timestamps[0]}, but the model was trained on data up to "
            f"{trained_until}. The data changed since training; retrain with `python -m training.run`."
        )


def backtest_split(
    model: nn.Module,
    split: SequenceSet,
    class_names: Sequence[str],
    backtest_cfg: BacktestConfig,
    periods: float,
    device: str = "cpu",
    calibration: Optional[SequenceSet] = None,
) -> BacktestResult:
    """Predict on a split and simulate trading the predictions.

    With ``backtest_cfg.exit_share`` set, the P(down) exit threshold is calibrated on
    the model's predictions for ``calibration``, which must come before ``split``.
    """
    if backtest_cfg.exit_share is not None:
        if calibration is None or len(calibration) == 0:
            raise ValueError("exit_share needs a non-empty calibration split before the traded period")
        calibration_probs, _ = predict_proba(model, make_loader(calibration, 1024, shuffle=False), device)
        threshold = calibrate_down_threshold(calibration_probs, class_names, backtest_cfg.exit_share)
        backtest_cfg = dataclasses.replace(backtest_cfg, rule="exit-on-down", down_prob_threshold=threshold)

    probs, _ = predict_proba(model, make_loader(split, 1024, shuffle=False), device)
    positions = positions_from_probs(
        probs,
        allow_short=backtest_cfg.allow_short,
        min_confidence=backtest_cfg.min_confidence,
        class_names=class_names,
        rule=backtest_cfg.rule,
        down_prob_threshold=backtest_cfg.down_prob_threshold,
    )
    result = run_backtest(positions, split.next_returns, backtest_cfg, periods, split.timestamps)
    if backtest_cfg.rule == "exit-on-down":
        result.down_prob_threshold = backtest_cfg.down_prob_threshold
    return result


def calibration_split(data: PreparedData, split_name: str) -> SequenceSet:
    """Data just before ``split_name`` for calibrating thresholds without looking ahead.

    For the test split this is the validation split. For the validation split it is the
    most recent training windows, as many as the validation split has.
    """
    if split_name == "test":
        return data.val
    return data.train.tail(len(data.val))


def buy_and_hold(split: SequenceSet, backtest_cfg: BacktestConfig, periods: float) -> BacktestResult:
    """Hold the asset for the whole split (one buy, one sell)."""
    return run_backtest(np.ones(len(split), dtype=np.int8), split.next_returns, backtest_cfg, periods, split.timestamps)


def backtest_model(
    model_name: str,
    raw: pd.DataFrame,
    split_name: str,
    backtest_cfg: BacktestConfig,
    checkpoint_dir: Path,
    device: str = "cpu",
    cache: Optional[dict[tuple, PreparedData]] = None,
) -> tuple[BacktestResult, SequenceSet]:
    """Rebuild one model's inputs from its checkpoint, predict and simulate trading."""
    data_cfg = get_data_config()
    model, meta = load_model(checkpoint_dir / f"{model_name}.pt", device)

    label_mode = meta.get("label_mode", "three_class")
    key = (meta["sequence_length"], meta["horizon"], meta["flat_threshold"], label_mode, tuple(meta["feature_columns"]))
    cache = {} if cache is None else cache
    if key not in cache:
        cache[key] = prepare_datasets(
            raw,
            sequence_length=meta["sequence_length"],
            train_split=data_cfg.train_split,
            val_split=data_cfg.val_split,
            timeframe=data_cfg.timeframe,
            horizon=meta["horizon"],
            flat_threshold=meta["flat_threshold"],
            feature_columns=meta["feature_columns"],
            scaler=FeatureScaler.from_dict(meta["scaler"]),
            label_mode=label_mode,
        )
    split: SequenceSet = getattr(cache[key], split_name)
    if len(split) == 0:
        raise ValueError(f"The {split_name} split is empty")
    check_no_overlap(split, meta, split_name)

    class_names = meta.get("class_names", CLASS_NAMES)
    result = backtest_split(
        model,
        split,
        class_names,
        backtest_cfg,
        periods_per_year(data_cfg.timeframe),
        device,
        calibration=calibration_split(cache[key], split_name),
    )
    return result, split


def describe_rule(backtest_cfg: BacktestConfig) -> str:
    """One-line description of how predictions become positions."""
    side = "long/short" if backtest_cfg.allow_short else "long only"
    if backtest_cfg.exit_share is not None:
        return (
            f"exit-on-down: long unless P(down) is in its top {backtest_cfg.exit_share:.0%}, "
            f"calibrated on data before the traded period, {side}"
        )
    if backtest_cfg.rule == "exit-on-down":
        return f"exit-on-down: long unless P(down) > {backtest_cfg.down_prob_threshold:.2f}, {side}"
    return f"argmax, {side}, min confidence {backtest_cfg.min_confidence:.2f}"


def run_tag(backtest_cfg: BacktestConfig) -> str:
    """Short name of the strategy settings, used in output file names."""
    if backtest_cfg.exit_share is not None:
        tag = f"exit_top{backtest_cfg.exit_share * 100:g}pct"
    elif backtest_cfg.rule == "exit-on-down":
        tag = f"exit_p{backtest_cfg.down_prob_threshold:g}"
    else:
        tag = "argmax" + (f"_conf{backtest_cfg.min_confidence:g}" if backtest_cfg.min_confidence else "")
    return tag + ("_short" if backtest_cfg.allow_short else "")


def format_threshold(result: BacktestResult) -> str:
    """The P(down) exit threshold a result used, or "-"."""
    return f"{result.down_prob_threshold:.3f}" if result.down_prob_threshold is not None else "-"


def format_results(results: dict[str, BacktestResult], split_name: str, backtest_cfg: BacktestConfig) -> str:
    """Render net-of-cost metrics for every strategy as a table."""
    header = (
        f"{'strategy':10s} {'return':>8s} {'annual':>8s} {'sharpe':>7s} {'max_dd':>8s} "
        f"{'exposure':>8s} {'trades':>7s} {'hit':>6s} {'costs':>7s} {'exit_at':>7s} | {'gross ret':>9s} {'gross sh':>8s}"
    )
    lines = [
        f"[{split_name}] after costs of {backtest_cfg.cost_per_trade:.2%} per trade ({describe_rule(backtest_cfg)})",
        header,
        "-" * len(header),
    ]
    for name, r in results.items():
        m, g = r.metrics, r.gross_metrics
        lines.append(
            f"{name:10s} {m.total_return:>+8.1%} {m.annual_return:>+8.1%} {m.sharpe:>7.2f} {m.max_drawdown:>8.1%} "
            f"{m.exposure:>8.0%} {m.trades:>7d} {m.hit_rate:>6.1%} {m.costs:>7.1%} {format_threshold(r):>7s} | "
            f"{g.total_return:>+9.1%} {g.sharpe:>8.2f}"
        )
    return "\n".join(lines)


def add_strategy_args(parser: argparse.ArgumentParser) -> None:
    """Command-line options that override the strategy settings from the environment."""
    parser.add_argument("--rule", choices=STRATEGY_RULES, help="how predictions become positions")
    parser.add_argument("--down-threshold", type=float, help="exit-on-down: P(down) above which to exit")
    parser.add_argument("--allow-short", action="store_true", default=None, help="go short instead of out on 'down'")
    parser.add_argument("--min-confidence", type=float, help="argmax: minimum predicted probability to act")
    parser.add_argument(
        "--exit-share", type=float, help="exit for the top share of P(down) (e.g. 0.1), calibrated before the traded period"
    )


def strategy_config(args: argparse.Namespace) -> BacktestConfig:
    """Backtest config from the environment, with command-line overrides applied."""
    cfg = get_backtest_config()
    overrides = {
        "rule": args.rule,
        "down_prob_threshold": args.down_threshold,
        "allow_short": args.allow_short,
        "min_confidence": args.min_confidence,
        "exit_share": args.exit_share,
    }
    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    if cfg.exit_share is not None:
        if not 0 < cfg.exit_share < 1:
            raise SystemExit("EXIT_SHARE must be between 0 and 1")
        cfg = dataclasses.replace(cfg, rule="exit-on-down")
    if cfg.rule not in STRATEGY_RULES:
        raise SystemExit(f"Unknown STRATEGY_RULE {cfg.rule!r}; expected one of {STRATEGY_RULES}")
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> dict[str, BacktestResult]:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, help="models to test (default: all trained)")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="data split to trade on")
    add_strategy_args(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data_cfg, train_cfg = get_data_config(), get_training_config()
    backtest_cfg = strategy_config(args)

    checkpoint_dir = train_cfg.checkpoint_dir
    models = args.models or [name for name in MODEL_NAMES if (checkpoint_dir / f"{name}.pt").exists()]
    if not models:
        raise SystemExit(f"No checkpoints found in {checkpoint_dir}/; run `python -m training.run` first")

    raw = CCXTLoader(
        data_cfg.exchange, data_cfg.symbol, data_cfg.timeframe, since=data_cfg.since, cache_dir=data_cfg.cache_dir
    ).load()

    cache: dict[tuple, PreparedData] = {}
    results: dict[str, BacktestResult] = {}
    for name in models:
        result, split = backtest_model(name, raw, args.split, backtest_cfg, checkpoint_dir, train_cfg.device, cache)
        if BUY_AND_HOLD not in results:
            results[BUY_AND_HOLD] = buy_and_hold(split, backtest_cfg, periods_per_year(data_cfg.timeframe))
            logger.info("Backtest period: %s -> %s (%d candles)", split.timestamps[0], split.timestamps[-1], len(split))
        results[name] = result

    print()
    print(format_results(results, args.split, backtest_cfg))

    equity = pd.DataFrame({name: r.frame["equity"] for name, r in results.items()})
    name = f"backtest_{args.split}_{run_tag(backtest_cfg)}"
    equity.to_csv(checkpoint_dir / f"{name}_equity.csv")
    summary = {
        "split": args.split,
        "config": dataclasses.asdict(backtest_cfg),
        "results": {
            strategy: {
                "net": dataclasses.asdict(r.metrics),
                "gross": dataclasses.asdict(r.gross_metrics),
                "down_prob_threshold": r.down_prob_threshold,
            }
            for strategy, r in results.items()
        },
    }
    (checkpoint_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nEquity curves and metrics saved in {checkpoint_dir}/{name}*")
    return results


if __name__ == "__main__":
    main()
