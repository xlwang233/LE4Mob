"""Stable, versioned location-encoder checkpoint format."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from le4mob.encoder import LocationEncoder, LocationEncoderConfig

CHECKPOINT_FORMAT = "le4mob.location-encoder"
CHECKPOINT_VERSION = 1
METADATA_SCHEMA = "le4mob.location-embeddings"
METADATA_VERSION = 2


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Coordinate and model metadata stored with a location encoder."""

    embedding_dim: int
    coordinate_order: str
    crs: str
    coordinate_units: str
    normalized: bool
    encoder_config: LocationEncoderConfig
    model_name: str = "LE4Mob"
    schema_name: str = METADATA_SCHEMA
    schema_version: int = METADATA_VERSION
    source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_name != METADATA_SCHEMA or self.schema_version != METADATA_VERSION:
            raise ValueError(
                f"Unsupported metadata schema: {self.schema_name!r} v{self.schema_version}."
            )
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.embedding_dim != self.encoder_config.output_dim:
            raise ValueError("embedding_dim must equal encoder_config.output_dim.")
        if self.coordinate_order not in {"lon-lat", "x-y"}:
            raise ValueError("coordinate_order must be 'lon-lat' or 'x-y'.")
        if not self.crs.strip() or not self.coordinate_units.strip():
            raise ValueError("crs and coordinate_units are required.")
        if self.coordinate_order == "lon-lat" and self.coordinate_units.lower() not in {
            "degree",
            "degrees",
        }:
            raise ValueError("lon-lat coordinates must declare degree units.")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["encoder_config"] = self.encoder_config.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> EmbeddingMetadata:
        copied = dict(values)
        if copied.get("schema_version", 1) == 1:
            legacy_orders = {"lon_lat": "lon-lat", "xy": "x-y"}
            order = copied.get("coordinate_order")
            copied["coordinate_order"] = legacy_orders.get(order, order)
            copied["schema_version"] = METADATA_VERSION
        copied["encoder_config"] = LocationEncoderConfig.from_dict(copied["encoder_config"])
        return cls(**copied)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's lowercase SHA-256 digest."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_json_data(value: Any, label: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON-serializable.") from error


def _atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_name)
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_weights_only(path: str | Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Torch versions before the weights_only argument
        return torch.load(path, map_location=map_location)


def save_location_encoder_checkpoint(
    path: str | Path,
    encoder: LocationEncoder,
    metadata: EmbeddingMetadata,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save an inference checkpoint independent of trainer internals."""
    if metadata.encoder_config != encoder.config:
        raise ValueError("Checkpoint metadata does not match the encoder configuration.")
    checkpoint_extra = dict(extra or {})
    _validate_json_data(metadata.to_dict(), "Checkpoint metadata")
    _validate_json_data(checkpoint_extra, "Checkpoint extra fields")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "encoder_config": encoder.config.to_dict(),
        "metadata": metadata.to_dict(),
        "state_dict": encoder.state_dict(),
        "extra": checkpoint_extra,
    }
    _atomic_torch_save(payload, path)


def load_location_encoder_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[LocationEncoder, EmbeddingMetadata, dict[str, Any]]:
    """Load and validate a public LE4Mob location-encoder checkpoint."""
    payload = _load_weights_only(path, map_location)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Not a supported LE4Mob location-encoder checkpoint.")

    metadata = EmbeddingMetadata.from_dict(payload["metadata"])
    if metadata.encoder_config.to_dict() != payload["encoder_config"]:
        raise ValueError("Checkpoint encoder configuration and metadata disagree.")
    encoder = LocationEncoder(metadata.encoder_config)
    encoder.load_state_dict(payload["state_dict"], strict=True)
    encoder.to(map_location)
    encoder.eval()
    return encoder, metadata, dict(payload.get("extra", {}))
