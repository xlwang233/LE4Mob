"""Versioned checkpoints shared by downstream mobility tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from le4mob.checkpoint import _atomic_torch_save, _load_weights_only, _validate_json_data

DOWNSTREAM_FORMAT = "le4mob.downstream"
DOWNSTREAM_VERSION = 2


@dataclass(frozen=True)
class DownstreamCheckpointMetadata:
    task: str
    model_name: str
    model_config: dict[str, Any]
    dataset: str
    data_split: str
    seed: int
    location_embedding_name: str
    location_checkpoint: str | None = None
    location_checkpoint_sha256: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task cannot be empty.")
        for field_name in (
            "model_name",
            "dataset",
            "data_split",
            "location_embedding_name",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} cannot be empty.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if (self.location_checkpoint is None) != (self.location_checkpoint_sha256 is None):
            raise ValueError("location_checkpoint and its SHA-256 checksum must be provided together.")
        checksum = self.location_checkpoint_sha256
        if checksum is not None and (
            len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError("location_checkpoint_sha256 must be a lowercase SHA-256 hex digest.")
        _validate_json_data(asdict(self), "Downstream checkpoint metadata")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DownstreamCheckpointMetadata:
        return cls(**dict(values))


@dataclass(frozen=True)
class LoadedDownstreamCheckpoint:
    metadata: DownstreamCheckpointMetadata
    epoch: int
    global_step: int
    optimizer_state_dict: dict[str, Any] | None
    external_state_prefixes: tuple[str, ...]


def _normalize_external_prefixes(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(prefixes)))
    if any(not prefix for prefix in normalized):
        raise ValueError("External state prefixes cannot be empty.")
    for index, prefix in enumerate(normalized):
        if any(other.startswith(prefix) for other in normalized[index + 1 :]):
            raise ValueError(f"Overlapping external state prefixes are not allowed: {normalized}.")
    return normalized


def _is_external(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def save_downstream_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: DownstreamCheckpointMetadata,
    *,
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    external_state_prefixes: tuple[str, ...] = (),
) -> None:
    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step cannot be negative.")
    external_state_prefixes = _normalize_external_prefixes(external_state_prefixes)
    if external_state_prefixes and metadata.location_checkpoint is None:
        raise ValueError("External model state requires a referenced location checkpoint.")
    complete_state = model.state_dict()
    for prefix in external_state_prefixes:
        if not any(key.startswith(prefix) for key in complete_state):
            raise ValueError(f"External state prefix does not match the model: {prefix!r}.")
    compact_state = {
        key: value for key, value in complete_state.items() if not _is_external(key, external_state_prefixes)
    }
    payload = {
        "format": DOWNSTREAM_FORMAT,
        "format_version": DOWNSTREAM_VERSION,
        "metadata": metadata.to_dict(),
        "state_dict": compact_state,
        "external_state_prefixes": list(external_state_prefixes),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    _atomic_torch_save(payload, path)


def load_downstream_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> LoadedDownstreamCheckpoint:
    """Load weights into a task model constructed from the stored model_config."""
    payload = _load_weights_only(path, map_location)
    version = payload.get("format_version")
    if payload.get("format") != DOWNSTREAM_FORMAT or version not in {1, DOWNSTREAM_VERSION}:
        raise ValueError("Not a supported LE4Mob downstream checkpoint.")
    external_state_prefixes = _normalize_external_prefixes(
        tuple(payload.get("external_state_prefixes", ())) if version >= 2 else ()
    )
    if not external_state_prefixes:
        model.load_state_dict(payload["state_dict"], strict=strict)
    else:
        model_keys = set(model.state_dict())
        external_keys = {key for key in model_keys if _is_external(key, external_state_prefixes)}
        if not external_keys:
            raise ValueError("Checkpoint external state prefixes do not match the constructed model.")
        expected_saved_keys = model_keys - external_keys
        actual_saved_keys = set(payload["state_dict"])
        if strict and actual_saved_keys != expected_saved_keys:
            missing = sorted(expected_saved_keys - actual_saved_keys)
            unexpected = sorted(actual_saved_keys - expected_saved_keys)
            raise ValueError(
                "Compact downstream state is incompatible with the model: "
                f"missing={missing}, unexpected={unexpected}."
            )
        incompatible = model.load_state_dict(payload["state_dict"], strict=False)
        missing_non_external = [
            key for key in incompatible.missing_keys if not _is_external(key, external_state_prefixes)
        ]
        if strict and (missing_non_external or incompatible.unexpected_keys):
            raise ValueError(
                "Could not restore compact downstream state: "
                f"missing={missing_non_external}, unexpected={incompatible.unexpected_keys}."
            )
    model.to(map_location)
    return LoadedDownstreamCheckpoint(
        metadata=DownstreamCheckpointMetadata.from_dict(payload["metadata"]),
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        optimizer_state_dict=payload.get("optimizer_state_dict"),
        external_state_prefixes=external_state_prefixes,
    )


def read_downstream_checkpoint_metadata(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedDownstreamCheckpoint:
    """Read checkpoint metadata and training counters without constructing its model."""
    payload = _load_weights_only(path, map_location)
    version = payload.get("format_version")
    if payload.get("format") != DOWNSTREAM_FORMAT or version not in {1, DOWNSTREAM_VERSION}:
        raise ValueError("Not a supported LE4Mob downstream checkpoint.")
    return LoadedDownstreamCheckpoint(
        metadata=DownstreamCheckpointMetadata.from_dict(payload["metadata"]),
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        optimizer_state_dict=payload.get("optimizer_state_dict"),
        external_state_prefixes=_normalize_external_prefixes(
            tuple(payload.get("external_state_prefixes", ())) if version >= 2 else ()
        ),
    )
