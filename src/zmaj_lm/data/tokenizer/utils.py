"""Utility functions for tokenizer training and evaluation."""

from typing import Any

from zmaj_lm.config.tokenizer_config import DatasetConfig


def build_load_args(dataset_config: DatasetConfig) -> dict[str, Any]:
    """Build arguments for load_dataset from a DatasetConfig.

    Args:
        dataset_config: Dataset configuration

    Returns:
        Dictionary of arguments for load_dataset
    """
    load_args: dict = {"path": dataset_config.path}
    if dataset_config.name is not None:
        load_args["name"] = dataset_config.name
    if dataset_config.trust_remote_code:
        load_args["trust_remote_code"] = True
    return load_args
