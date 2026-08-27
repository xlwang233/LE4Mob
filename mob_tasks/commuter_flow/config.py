"""Configuration helpers shared by the commuter flow notebook and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Commuter flow configuration must be a YAML mapping.")

    required_sections = {"experiment", "data", "model", "training", "output", "debug"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Missing commuter flow configuration sections: {sorted(missing)}")

    split_ids = config["data"].get("split_ids")
    if split_ids != [0, 1, 2, 3, 4]:
        raise ValueError("The London experiment must retain geographic splits [0, 1, 2, 3, 4].")
    if config["model"].get("type") not in {"deepgravity", "bilinear"}:
        raise ValueError("model.type must be 'deepgravity' or 'bilinear'.")
    if config["data"].get("distance_column") not in {"distances", "distance_normed"}:
        raise ValueError("data.distance_column must be 'distances' or 'distance_normed'.")
    if not config["data"].get("representations"):
        raise ValueError("At least one location representation must be configured.")
    return config


def resolve_model_seeds(training_config: dict[str, Any]) -> list[int]:
    """Select the paper's repeated seeds or the fixed single-run seed."""
    if training_config.get("use_multiple_model_seeds", True):
        seeds = [int(seed) for seed in training_config.get("model_seeds", [])]
        if not seeds:
            raise ValueError("model_seeds cannot be empty when repeated model seeds are enabled.")
        return seeds
    return [10]
