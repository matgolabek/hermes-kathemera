"""Train and compare models on the configured market.

Usage::

    python -m training.run                    # every model: baselines, LSTM, GRU
    python -m training.run --models lstm gru  # a subset
    python -m training.run --eval-test        # also score the held-out test split

Each model is trained with the same data, seed and settings, its best epoch (lowest
validation loss) is saved to ``CHECKPOINT_DIR/<model>.pt``, and a comparison table
is printed. The test split is only scored with ``--eval-test``: use it once, for the
final choice, or it stops being an honest estimate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from typing import Any, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig, get_data_config, get_model_config, get_training_config
from data_pipeline.loaders import CCXTLoader
from data_pipeline.pipeline import PreparedData, SequenceSet, prepare_datasets
from models import MODEL_NAMES, build_model
from training.checkpoint import load_model
from training.dataset import TimeSeriesDataset
from training.metrics import class_prior, classification_metrics
from training.train import predict_proba, set_seed, train_model

logger = logging.getLogger(__name__)


def make_loader(split: SequenceSet, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    """Wrap a split in a DataLoader; shuffling is seeded so runs are repeatable."""
    dataset = TimeSeriesDataset(torch.from_numpy(split.X), torch.from_numpy(split.y))
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_and_evaluate(
    data: PreparedData, model_cfg: ModelConfig, train_cfg: TrainingConfig, eval_test: bool = False
) -> dict[str, Any]:
    """Train one model, reload its best checkpoint and score it on validation (and test)."""
    for name, split in (("train", data.train), ("val", data.val)):
        if len(split) == 0:
            raise ValueError(f"The {name} split is empty; load more data or shorten SEQUENCE_LENGTH")

    set_seed(train_cfg.seed)
    model = build_model(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    checkpoint_path = train_cfg.checkpoint_dir / f"{model_cfg.model_name}.pt"
    metadata = {
        "model_config": dataclasses.asdict(model_cfg),
        **data.metadata(),
        "training": {k: str(v) if not isinstance(v, (int, float)) else v for k, v in dataclasses.asdict(train_cfg).items()},
        "train_period": [str(data.train.timestamps[0]), str(data.train.timestamps[-1])],
    }

    history = train_model(
        model,
        make_loader(data.train, train_cfg.batch_size, shuffle=True, seed=train_cfg.seed),
        make_loader(data.val, train_cfg.batch_size, shuffle=False),
        nn.CrossEntropyLoss(),
        optimizer,
        train_cfg.epochs,
        train_cfg.device,
        checkpoint_path,
        metadata=metadata,
        patience=train_cfg.patience or None,
        grad_clip=train_cfg.grad_clip or None,
    )

    best_model, _ = load_model(checkpoint_path, train_cfg.device)
    prior = class_prior(data.train.y, len(data.class_names))
    splits = [("val", data.val)] + ([("test", data.test)] if eval_test else [])
    result: dict[str, Any] = {
        "model": model_cfg.model_name,
        "parameters": sum(p.numel() for p in best_model.parameters()),
        "best_epoch": history.best_epoch,
        "epochs_run": len(history.val),
        "checkpoint": str(checkpoint_path),
        "class_names": data.class_names,
    }
    for name, split in splits:
        if len(split) == 0:
            logger.warning("The %s split is empty; skipping its metrics", name)
            continue
        probs, labels = predict_proba(best_model, make_loader(split, train_cfg.batch_size, shuffle=False), train_cfg.device)
        result[name] = classification_metrics(probs, labels, prior, data.class_names, split.returns, data.horizon)
    return result


def format_results(results: Sequence[dict[str, Any]], split: str) -> str:
    """Render one split's metrics as a comparison table."""
    names = next((r["class_names"] for r in results if split in r), [])
    header = (
        f"{'model':10s} {'params':>9s} {'epoch':>5s} {'log_loss':>9s} {'skill':>8s} {'acc':>6s} {'bal_acc':>7s} "
        f"{'dir_ic':>7s} {'vol_ic':>7s} {'dir_acc':>7s}  predicted {'/'.join(names)}"
    )
    lines = [f"[{split}] baseline: always predicting training class frequencies (skill 0)", header, "-" * len(header)]
    for r in results:
        m = r.get(split)
        if m is None:
            continue
        shares = "/".join(f"{m[f'pred_{c}']:.0%}" for c in r["class_names"])
        dir_acc = f"{m['direction_accuracy']:.3f}" if m.get("direction_accuracy") is not None else "-"
        lines.append(
            f"{r['model']:10s} {r['parameters']:>9,d} {r['best_epoch']:>5d} {m['log_loss']:>9.4f} "
            f"{m['skill']:>+8.2%} {m['accuracy']:>6.3f} {m['balanced_accuracy']:>7.3f} "
            f"{m['direction_ic']:>+7.3f} {m['volatility_ic']:>+7.3f} {dir_acc:>7s}  {shares}"
        )
    first = next((r[split] for r in results if split in r), None)
    if first is not None:
        lines.append(
            f"(prior log_loss {first['prior_log_loss']:.4f}; majority-class accuracy {first['majority_accuracy']:.3f}; "
            f"|ic| below {first['ic_noise']:.3f} is likely noise)"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES, help="models to train")
    parser.add_argument("--eval-test", action="store_true", help="also score the held-out test split")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data_cfg, model_cfg, train_cfg = get_data_config(), get_model_config(), get_training_config()

    raw = CCXTLoader(
        data_cfg.exchange, data_cfg.symbol, data_cfg.timeframe, since=data_cfg.since, cache_dir=data_cfg.cache_dir
    ).load()
    data = prepare_datasets(
        raw,
        sequence_length=data_cfg.sequence_length,
        train_split=data_cfg.train_split,
        val_split=data_cfg.val_split,
        timeframe=data_cfg.timeframe,
        horizon=data_cfg.horizon,
        flat_threshold=data_cfg.flat_threshold,
        label_mode=data_cfg.label_mode,
    )
    model_cfg = dataclasses.replace(model_cfg, input_size=len(data.feature_columns), output_size=len(data.class_names))
    logger.info("Samples: train=%d val=%d test=%d", len(data.train), len(data.val), len(data.test))
    logger.info("Train classes: %s | val classes: %s", data.train.class_counts(), data.val.class_counts())

    results = []
    for name in args.models:
        logger.info("=== Training %s", name)
        results.append(train_and_evaluate(data, dataclasses.replace(model_cfg, model_name=name), train_cfg, args.eval_test))

    print()
    print(format_results(results, "val"))
    if args.eval_test:
        print()
        print(format_results(results, "test"))

    results_path = train_cfg.checkpoint_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nCheckpoints and results.json saved in {train_cfg.checkpoint_dir}/")
    return results


if __name__ == "__main__":
    main()
