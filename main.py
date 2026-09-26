"""Example pipeline entry point for model training and deployment wiring."""

from __future__ import annotations

import logging

from config import get_data_config, get_execution_config, get_model_config, get_training_config
from data_pipeline.loaders import CCXTLoader
from data_pipeline.pipeline import prepare_datasets
from data_pipeline.preprocess import class_names_for
from models import build_model


def main() -> None:
    """Load configs, download market data, build model-ready datasets and the model."""
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
        horizon=data_cfg.horizon,
        flat_threshold=data_cfg.flat_threshold,
        label_mode=data_cfg.label_mode,
    )

    if len(data.feature_columns) != model_cfg.input_size:
        raise ValueError(f"INPUT_SIZE={model_cfg.input_size} but the pipeline produces {len(data.feature_columns)} features")
    class_names = class_names_for(data_cfg.label_mode)
    if model_cfg.output_size != len(class_names):
        raise ValueError(f"OUTPUT_SIZE={model_cfg.output_size} but there are {len(class_names)} classes {class_names}")

    print("Datasets:")
    for name, split in (("train", data.train), ("val", data.val), ("test", data.test)):
        if len(split):
            print(f"  {name:5s} X={split.X.shape} {split.timestamps[0]} -> {split.timestamps[-1]} classes={split.class_counts()}")
        else:
            print(f"  {name:5s} empty")

    model = build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_cfg.model_name} with {n_params:,} parameters")


if __name__ == "__main__":
    main()
