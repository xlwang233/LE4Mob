"""Typed configuration for framework-neutral LE4Mob pre-training runs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from le4mob.losses import DistanceRegularizationConfig
from le4mob.model import LE4MobModelConfig


def _merge_mappings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge one setting profile into the shared YAML configuration."""
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@dataclass(frozen=True)
class PretrainingExperimentConfig:
    name: str
    dataset: str
    setting: str
    heldout_policy: str = "none"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.dataset.strip():
            raise ValueError("Experiment name and dataset cannot be empty.")
        if self.setting not in {"conventional", "inductive"}:
            raise ValueError("setting must be 'conventional' or 'inductive'.")
        if self.setting == "conventional" and self.heldout_policy != "none":
            raise ValueError("A conventional experiment must use heldout_policy='none'.")
        if self.setting == "inductive" and self.heldout_policy == "none":
            raise ValueError("An inductive experiment must declare its heldout_policy.")


@dataclass(frozen=True)
class PretrainingDataConfig:
    path: str
    id_column: str
    text_column: str
    coordinate_columns: tuple[str, str]
    coordinate_order: str
    crs: str
    coordinate_units: str
    validation_fraction: float = 0.1
    split_seed: int = 42
    max_token_length: int | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("data.path cannot be empty.")
        if not self.id_column.strip() or not self.text_column.strip():
            raise ValueError("ID and text column names cannot be empty.")
        if len(self.coordinate_columns) != 2 or any(not column.strip() for column in self.coordinate_columns):
            raise ValueError("coordinate_columns must contain exactly two column names.")
        if self.coordinate_order not in {"lon-lat", "x-y"}:
            raise ValueError("coordinate_order must be 'lon-lat' or 'x-y'.")
        if not self.crs.strip() or not self.coordinate_units.strip():
            raise ValueError("CRS and coordinate units are required.")
        if self.coordinate_order == "lon-lat" and self.coordinate_units.lower() not in {
            "degree",
            "degrees",
        }:
            raise ValueError("lon-lat coordinates must use degree units.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.split_seed < 0:
            raise ValueError("split_seed cannot be negative.")
        if self.max_token_length is not None and self.max_token_length <= 0:
            raise ValueError("max_token_length must be positive when provided.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> PretrainingDataConfig:
        copied = dict(values)
        copied["coordinate_columns"] = tuple(copied["coordinate_columns"])
        return cls(**copied)


@dataclass(frozen=True)
class PretrainingTrainingConfig:
    seed: int = 42
    device: str = "cuda:0"
    batch_size: int = 128
    max_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    num_workers: int = 0
    log_every_steps: int = 100
    gradient_clip_norm: float | None = None
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if self.batch_size <= 1:
            raise ValueError("batch_size must be greater than one for contrastive pre-training.")
        if self.max_epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("max_epochs and learning_rate must be positive.")
        if self.weight_decay < 0 or self.num_workers < 0:
            raise ValueError("weight_decay and num_workers cannot be negative.")
        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided.")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive when provided.")


@dataclass(frozen=True)
class PretrainingOutputConfig:
    directory: str

    def __post_init__(self) -> None:
        if not self.directory.strip():
            raise ValueError("output.directory cannot be empty.")


@dataclass(frozen=True)
class PretrainingRunConfig:
    experiment: PretrainingExperimentConfig
    data: PretrainingDataConfig
    model: LE4MobModelConfig
    distance_regularization: DistanceRegularizationConfig
    training: PretrainingTrainingConfig
    output: PretrainingOutputConfig

    def __post_init__(self) -> None:
        expected_system = "lon-lat" if self.data.coordinate_order == "lon-lat" else "projected"
        if self.distance_regularization.coordinate_system != expected_system:
            raise ValueError(
                "distance_regularization.coordinate_system must agree with data.coordinate_order."
            )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["data"]["coordinate_columns"] = list(self.data.coordinate_columns)
        values["model"] = self.model.to_dict()
        values["distance_regularization"] = self.distance_regularization.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> PretrainingRunConfig:
        copied = dict(values)
        return cls(
            experiment=PretrainingExperimentConfig(**copied["experiment"]),
            data=PretrainingDataConfig.from_dict(copied["data"]),
            model=LE4MobModelConfig.from_dict(copied["model"]),
            distance_regularization=DistanceRegularizationConfig.from_dict(
                copied["distance_regularization"]
            ),
            training=PretrainingTrainingConfig(**copied["training"]),
            output=PretrainingOutputConfig(**copied["output"]),
        )

    @classmethod
    def from_yaml(cls, path: str | Path, *, setting: str | None = None) -> PretrainingRunConfig:
        with open(path) as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            raise ValueError("Pre-training config must be a YAML mapping.")
        values = dict(values)
        profiles = values.pop("settings", None)
        if profiles is None:
            config = cls.from_dict(values)
            if setting is not None and config.experiment.setting != setting:
                raise ValueError(
                    f"Requested setting {setting!r}, but the config declares "
                    f"{config.experiment.setting!r}."
                )
            return config
        if not isinstance(profiles, Mapping):
            raise ValueError("Pre-training config settings must be a YAML mapping.")
        if setting is None:
            raise ValueError("A setting is required for a config containing setting profiles.")
        if setting not in profiles:
            raise ValueError(f"Setting {setting!r} is not defined; choose one of {sorted(profiles)}.")
        profile = profiles[setting]
        if not isinstance(profile, Mapping):
            raise ValueError(f"The {setting!r} setting profile must be a YAML mapping.")

        resolved = _merge_mappings(values, profile)
        experiment = dict(resolved.get("experiment", {}))
        declared_setting = experiment.get("setting")
        if declared_setting is not None and declared_setting != setting:
            raise ValueError(
                f"The {setting!r} profile declares conflicting setting {declared_setting!r}."
            )
        dataset = experiment.get("dataset")
        if dataset:
            experiment.setdefault("name", f"{dataset}_{setting}")
        experiment["setting"] = setting
        resolved["experiment"] = experiment
        return cls.from_dict(resolved)
