"""Compact LE4Mob training checkpoints and inference-checkpoint extraction."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from le4mob.checkpoint import (
    EmbeddingMetadata,
    _atomic_torch_save,
    _load_weights_only,
    _validate_json_data,
    save_location_encoder_checkpoint,
)
from le4mob.encoder import LocationEncoder
from le4mob.losses import DistanceRegularizationConfig, LE4MobLoss
from le4mob.model import LE4MobModel, LE4MobModelConfig, LE4MobPretrainingModule

PRETRAINING_FORMAT = "le4mob.pretraining"
PRETRAINING_VERSION = 2

_SAVED_MODEL_PREFIXES = (
    "model.location_encoder.",
    "model.text_projection.",
)
_SAVED_MODEL_KEYS = {"model.logit_scale"}


@dataclass(frozen=True)
class LoadedPretrainingCheckpoint:
    module: LE4MobPretrainingModule
    metadata: EmbeddingMetadata
    epoch: int
    global_step: int
    optimizer_state_dict: dict[str, Any] | None
    scheduler_state_dict: dict[str, Any] | None
    random_state: dict[str, Any]
    provenance: dict[str, Any]
    text_encoder: dict[str, Any]


def describe_text_encoder(model: LE4MobModel) -> dict[str, Any]:
    """Describe the external frozen language model without serializing it."""
    config = getattr(model.text_encoder, "config", None)
    resolved_revision = getattr(config, "_commit_hash", None)
    try:
        from importlib.metadata import version

        transformers_version = version("transformers")
    except Exception:  # pragma: no cover - only relevant in unusual source installs
        transformers_version = None
    return {
        "name": model.config.text_encoder_name,
        "requested_revision": model.config.text_encoder_revision,
        "resolved_revision": resolved_revision or model.config.text_encoder_revision,
        "library": "transformers",
        "library_version": transformers_version,
        "frozen": model.config.freeze_text_encoder,
    }


def _compact_state_dict(module: LE4MobPretrainingModule) -> dict[str, torch.Tensor]:
    """Return all state needed for learned components, excluding the frozen LM."""
    return {
        name: value
        for name, value in module.state_dict().items()
        if name in _SAVED_MODEL_KEYS or name.startswith(_SAVED_MODEL_PREFIXES)
    }


def _load_compact_state_dict(
    module: LE4MobPretrainingModule,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    expected = set(_compact_state_dict(module))
    actual = set(state_dict)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "Compact pre-training state is incompatible with the model: "
            f"missing={missing}, unexpected={unexpected}."
        )
    incompatible = module.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_language_missing = [
        key for key in incompatible.missing_keys if not key.startswith("model.text_encoder.")
    ]
    if unexpected or non_language_missing:
        raise ValueError(
            "Could not restore compact pre-training state: "
            f"missing={non_language_missing}, unexpected={unexpected}."
        )


def capture_random_state() -> dict[str, Any]:
    """Capture Python, NumPy, and Torch RNG state in a weights-only-safe form."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state after loading a checkpoint for an exact training resume."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_pretraining_checkpoint(
    path: str | Path,
    module: LE4MobPretrainingModule,
    metadata: EmbeddingMetadata,
    *,
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Save resumable training state while keeping the frozen language model external."""
    model_config = module.model.config
    if not model_config.freeze_text_encoder or any(
        parameter.requires_grad for parameter in module.model.text_encoder.parameters()
    ):
        raise ValueError(
            "Compact checkpoints require a frozen text encoder; trainable language-model "
            "weights cannot be omitted."
        )
    if metadata.encoder_config != model_config.location_encoder:
        raise ValueError("Pre-training metadata does not match the model's location encoder.")
    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step cannot be negative.")
    provenance_dict = dict(provenance or {})
    required_provenance = {"dataset", "split", "seed"}
    missing_provenance = sorted(required_provenance - set(provenance_dict))
    if missing_provenance:
        raise ValueError(f"Missing required pre-training provenance: {missing_provenance}.")
    _validate_json_data(metadata.to_dict(), "Pre-training metadata")
    _validate_json_data(provenance_dict, "Pre-training provenance")
    text_encoder_metadata = describe_text_encoder(module.model)
    _validate_json_data(text_encoder_metadata, "Text-encoder metadata")

    payload = {
        "format": PRETRAINING_FORMAT,
        "format_version": PRETRAINING_VERSION,
        "model_config": model_config.to_dict(),
        "text_encoder": text_encoder_metadata,
        "distance_config": module.objective.distance_config.to_dict(),
        "metadata": metadata.to_dict(),
        "trainable_state_dict": _compact_state_dict(module),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "random_state": capture_random_state(),
        "provenance": provenance_dict,
    }
    _atomic_torch_save(payload, path)


def load_pretraining_checkpoint(
    path: str | Path,
    *,
    text_encoder: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedPretrainingCheckpoint:
    """Restore compact training state, resolving the frozen language model externally."""
    payload = _load_weights_only(path, map_location)
    if payload.get("format") != PRETRAINING_FORMAT or payload.get("format_version") != PRETRAINING_VERSION:
        raise ValueError("Not a supported LE4Mob pre-training checkpoint.")

    model_config = LE4MobModelConfig.from_dict(payload["model_config"])
    distance_config = DistanceRegularizationConfig.from_dict(payload["distance_config"])
    metadata = EmbeddingMetadata.from_dict(payload["metadata"])
    if metadata.encoder_config != model_config.location_encoder:
        raise ValueError("Pre-training checkpoint configuration and metadata disagree.")

    text_encoder_metadata = dict(payload["text_encoder"])
    if text_encoder_metadata.get("name") != model_config.text_encoder_name:
        raise ValueError("Checkpoint text-encoder metadata and model configuration disagree.")
    if text_encoder is None:
        from transformers import AutoModel

        revision = text_encoder_metadata.get("resolved_revision") or model_config.text_encoder_revision
        text_encoder = AutoModel.from_pretrained(model_config.text_encoder_name, revision=revision)

    module = LE4MobPretrainingModule(
        LE4MobModel(model_config, text_encoder=text_encoder),
        LE4MobLoss(distance_config),
    )
    _load_compact_state_dict(module, payload["trainable_state_dict"])
    module.to(map_location)
    return LoadedPretrainingCheckpoint(
        module=module,
        metadata=metadata,
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        optimizer_state_dict=payload.get("optimizer_state_dict"),
        scheduler_state_dict=payload.get("scheduler_state_dict"),
        random_state=dict(payload["random_state"]),
        provenance=dict(payload.get("provenance", {})),
        text_encoder=text_encoder_metadata,
    )


def extract_location_encoder_checkpoint(
    pretraining_path: str | Path,
    output_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> EmbeddingMetadata:
    """Create a small downstream inference checkpoint without loading the language model."""
    payload = _load_weights_only(pretraining_path, map_location)
    if payload.get("format") != PRETRAINING_FORMAT or payload.get("format_version") != PRETRAINING_VERSION:
        raise ValueError("Not a supported LE4Mob pre-training checkpoint.")

    metadata = EmbeddingMetadata.from_dict(payload["metadata"])
    encoder = LocationEncoder(metadata.encoder_config)
    prefix = "model.location_encoder."
    location_state = {
        key[len(prefix) :]: value
        for key, value in payload["trainable_state_dict"].items()
        if key.startswith(prefix)
    }
    encoder.load_state_dict(location_state, strict=True)
    save_location_encoder_checkpoint(
        output_path,
        encoder,
        metadata,
        extra={
            "source_pretraining_checkpoint": Path(pretraining_path).name,
            "source_epoch": int(payload["epoch"]),
            "source_global_step": int(payload["global_step"]),
        },
    )
    return metadata
