"""Example pipeline entry point for model training and deployment wiring."""

from __future__ import annotations

import logging

from config import get_data_config, get_execution_config, get_model_config, get_training_config
from data_pipeline.loaders import CCXTLoader
from data_pipeline.pipeline import prepare_datasets


def main() -> None:
    """Load configs, download market data and build model-ready datasets."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data_cfg = get_data_config()
    model_cfg = get_model_config()
    train_cfg = get_training_config()
    exec_cfg = get_execution_config()

    print("Loaded configs:")
    print(data_cfg)
    print(model_cfg)
    print(train_cfg)
    print(exec_cfg)

    loader = CCXTLoader(
        data_cfg.exchange, data_cfg.symbol, data_cfg.timeframe, since=data_cfg.since, cache_dir=data_cfg.cache_dir
    )
    data = prepare_datasets(
        loader.load(),
        sequence_length=data_cfg.sequence_length,
        train_split=data_cfg.train_split,
        val_split=data_cfg.val_split,
        timeframe=data_cfg.timeframe,
    )

    if len(data.feature_columns) != model_cfg.input_size:
        raise ValueError(f"INPUT_SIZE={model_cfg.input_size} but the pipeline produces {len(data.feature_columns)} features")

    print("Datasets:")
    for name, split in (("train", data.train), ("val", data.val), ("test", data.test)):
        if len(split):
            print(f"  {name:5s} X={split.X.shape} y={split.y.shape} {split.timestamps[0]} -> {split.timestamps[-1]}")
        else:
            print(f"  {name:5s} empty")


if __name__ == "__main__":
    main()
