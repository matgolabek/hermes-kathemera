"""Example pipeline entry point for model training and deployment wiring."""

from __future__ import annotations

from config import get_data_config, get_execution_config, get_model_config, get_training_config


def main() -> None:
    """Load configs and prepare the project runtime."""
    data_cfg = get_data_config()
    model_cfg = get_model_config()
    train_cfg = get_training_config()
    exec_cfg = get_execution_config()

    print("Loaded configs:")
    print(data_cfg)
    print(model_cfg)
    print(train_cfg)
    print(exec_cfg)


if __name__ == "__main__":
    main()
