"""Framework-neutral training loop for LE4Mob pre-training."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from le4mob.checkpoint import EmbeddingMetadata, sha256_file
from le4mob.losses import LE4MobLoss, LossOutput
from le4mob.model import LE4MobModel, LE4MobPretrainingModule
from le4mob.pretrain.checkpoint import (
    describe_text_encoder,
    extract_location_encoder_checkpoint,
    load_pretraining_checkpoint,
    restore_random_state,
    save_pretraining_checkpoint,
)
from le4mob.pretrain.config import PretrainingRunConfig
from le4mob.pretrain.data import PretrainingDataLoaders, build_pretraining_dataloaders


@dataclass(frozen=True)
class EpochMetrics:
    total: float
    contrastive: float
    distance: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "contrastive": self.contrastive,
            "distance": self.distance,
        }


@dataclass(frozen=True)
class PretrainingResult:
    training_checkpoint: Path
    location_checkpoint: Path
    last_checkpoint: Path
    history_path: Path
    metadata_path: Path
    best_epoch: int
    best_validation_loss: float
    global_step: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def _load_tokenizer(name: str, revision: str | None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def _build_optimizer(module: LE4MobPretrainingModule, learning_rate: float, weight_decay: float):
    """Match the AdamW parameter grouping used by the research implementation."""

    def exclude(name: str, parameter: Tensor) -> bool:
        return parameter.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or "logit_scale" in name

    named_parameters = [
        (name, parameter) for name, parameter in module.model.named_parameters() if parameter.requires_grad
    ]
    gain_or_bias = [parameter for name, parameter in named_parameters if exclude(name, parameter)]
    remaining = [parameter for name, parameter in named_parameters if not exclude(name, parameter)]
    return torch.optim.AdamW(
        [
            {"params": gain_or_bias, "weight_decay": 0.0},
            {"params": remaining, "weight_decay": weight_decay},
        ],
        lr=learning_rate,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "coordinates": batch["coordinates"].to(device, non_blocking=True),
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
    }


def _accumulate(sums: dict[str, float], loss: LossOutput, batch_size: int) -> None:
    sums["total"] += float(loss.total.detach()) * batch_size
    sums["contrastive"] += float(loss.contrastive.detach()) * batch_size
    sums["distance"] += float(loss.distance.detach()) * batch_size
    sums["samples"] += batch_size


def _finalize_metrics(sums: dict[str, float]) -> EpochMetrics:
    if sums["samples"] == 0:
        raise ValueError("A pre-training data loader produced no samples.")
    return EpochMetrics(
        total=sums["total"] / sums["samples"],
        contrastive=sums["contrastive"] / sums["samples"],
        distance=sums["distance"] / sums["samples"],
    )


def _train_epoch(
    module: LE4MobPretrainingModule,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: float | None,
    log_every_steps: int,
    global_step: int,
) -> tuple[EpochMetrics, int]:
    module.train()
    sums = {"total": 0.0, "contrastive": 0.0, "distance": 0.0, "samples": 0.0}
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = module(**batch)
        loss.total.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_norm)
        optimizer.step()
        global_step += 1
        _accumulate(sums, loss, batch["coordinates"].shape[0])
        if batch_index % log_every_steps == 0:
            current = _finalize_metrics(sums)
            print(
                f"  step {batch_index}/{len(loader)}: total={current.total:.5f}, "
                f"contrastive={current.contrastive:.5f}, distance={current.distance:.5f}"
            )
    return _finalize_metrics(sums), global_step


def _validate_epoch(
    module: LE4MobPretrainingModule,
    loader,
    device: torch.device,
) -> EpochMetrics:
    module.eval()
    sums = {"total": 0.0, "contrastive": 0.0, "distance": 0.0, "samples": 0.0}
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            loss = module(**batch)
            _accumulate(sums, loss, batch["coordinates"].shape[0])
    return _finalize_metrics(sums)


def _portable_path(path: Path) -> str:
    """Keep repository-relative paths and reduce external paths to their filename."""
    resolved_path = path.expanduser().resolve()
    try:
        return resolved_path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved_path.name


def _metadata(config: PretrainingRunConfig, data_path: Path) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        embedding_dim=config.model.location_encoder.output_dim,
        coordinate_order=config.data.coordinate_order,
        crs=config.data.crs,
        coordinate_units=config.data.coordinate_units,
        normalized=False,
        encoder_config=config.model.location_encoder,
        source=_portable_path(data_path),
        extra={
            "dataset": config.experiment.dataset,
            "setting": config.experiment.setting,
            "heldout_policy": config.experiment.heldout_policy,
        },
    )


def _base_provenance(
    config: PretrainingRunConfig,
    data_path: Path,
    data: PretrainingDataLoaders,
) -> dict[str, Any]:
    return {
        "dataset": config.experiment.dataset,
        "split": config.experiment.setting,
        "seed": config.training.seed,
        "experiment_name": config.experiment.name,
        "heldout_policy": config.experiment.heldout_policy,
        "data_path": _portable_path(data_path),
        "data_sha256": sha256_file(data_path),
        "data_rows": len(data.dataset),
        "train_rows": data.train_size,
        "validation_rows": data.validation_size,
        "validation_fraction": config.data.validation_fraction,
        "split_seed": config.data.split_seed,
        "coordinate_columns": list(config.data.coordinate_columns),
        "coordinate_order": config.data.coordinate_order,
        "crs": config.data.crs,
        "coordinate_units": config.data.coordinate_units,
    }


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(history, indent=2) + "\n")


def _write_run_metadata(
    path: Path,
    *,
    config: PretrainingRunConfig,
    text_encoder: Mapping[str, Any],
    provenance: Mapping[str, Any],
    best_epoch: int,
    best_validation_loss: float,
) -> None:
    payload = {
        "format": "le4mob.pretraining_metadata",
        "format_version": 1,
        "text_encoder": dict(text_encoder),
        "model": config.model.to_dict(),
        "distance_regularization": config.distance_regularization.to_dict(),
        "provenance": dict(provenance),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "artifacts": {
            "pretraining_checkpoint": "pretraining_checkpoint.pt",
            "location_encoder": "location_encoder.pt",
            "latest_resume_checkpoint": "last_checkpoint.pt",
            "history": "history.json",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_pretraining(
    config: PretrainingRunConfig,
    *,
    data_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    device: str | None = None,
    resume_from: str | Path | None = None,
    tokenizer: Any | None = None,
    text_encoder: nn.Module | None = None,
) -> PretrainingResult:
    """Run LE4Mob pre-training and save compact training and location checkpoints."""
    if not config.model.freeze_text_encoder:
        raise ValueError("Native compact checkpointing requires model.freeze_text_encoder=true.")
    if data_path is not None:
        config = replace(config, data=replace(config.data, path=str(data_path)))
    if output_directory is not None:
        config = replace(config, output=replace(config.output, directory=str(output_directory)))
    if device is not None:
        config = replace(config, training=replace(config.training, device=device))

    resolved_data_path = Path(config.data.path).expanduser().resolve()
    output_root = Path(config.output.directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    training_checkpoint = output_root / "pretraining_checkpoint.pt"
    location_checkpoint = output_root / "location_encoder.pt"
    last_checkpoint = output_root / "last_checkpoint.pt"
    history_path = output_root / "history.json"
    metadata_path = output_root / "pretraining_metadata.json"

    _set_seed(config.training.seed)
    training_device = _resolve_device(config.training.device)
    tokenizer = tokenizer or _load_tokenizer(
        config.model.text_encoder_name,
        config.model.text_encoder_revision,
    )
    data = build_pretraining_dataloaders(
        replace(config.data, path=str(resolved_data_path)),
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        training_seed=config.training.seed,
        pin_memory=training_device.type == "cuda",
    )
    metadata = _metadata(config, resolved_data_path)
    provenance = _base_provenance(config, resolved_data_path, data)

    start_epoch = 1
    global_step = 0
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if resume_from is None:
        module = LE4MobPretrainingModule(
            LE4MobModel(config.model, text_encoder=text_encoder),
            LE4MobLoss(config.distance_regularization),
        ).to(training_device)
    else:
        loaded = load_pretraining_checkpoint(
            resume_from,
            text_encoder=text_encoder,
            map_location=training_device,
        )
        if loaded.module.model.config != config.model:
            raise ValueError("Resume checkpoint model config does not match the run config.")
        if loaded.module.objective.distance_config != config.distance_regularization:
            raise ValueError("Resume checkpoint distance config does not match the run config.")
        if loaded.metadata != metadata:
            raise ValueError("Resume checkpoint metadata does not match the run config.")
        if loaded.provenance.get("data_sha256") != provenance["data_sha256"]:
            raise ValueError("Resume checkpoint was created from a different POI table.")
        module = loaded.module
        start_epoch = loaded.epoch + 1
        global_step = loaded.global_step
        best_epoch = int(loaded.provenance.get("best_epoch", loaded.epoch))
        best_validation_loss = float(loaded.provenance.get("best_validation_loss", float("inf")))
        epochs_without_improvement = int(loaded.provenance.get("epochs_without_improvement", 0))
        if history_path.is_file():
            history = json.loads(history_path.read_text())

    optimizer = _build_optimizer(
        module,
        config.training.learning_rate,
        config.training.weight_decay,
    )
    if resume_from is not None:
        if loaded.optimizer_state_dict is None:
            raise ValueError("Resume checkpoint does not contain optimizer state.")
        optimizer.load_state_dict(loaded.optimizer_state_dict)
        restore_random_state(loaded.random_state)

    print(
        f"Pre-training {config.experiment.name} on {training_device}: "
        f"{data.train_size} train / {data.validation_size} validation POIs"
    )
    for epoch in range(start_epoch, config.training.max_epochs + 1):
        epoch_start = time.time()
        data.train_sampler.set_epoch(epoch)
        train_metrics, global_step = _train_epoch(
            module,
            data.train,
            optimizer,
            training_device,
            gradient_clip_norm=config.training.gradient_clip_norm,
            log_every_steps=config.training.log_every_steps,
            global_step=global_step,
        )
        validation_metrics = _validate_epoch(module, data.validation, training_device)
        improved = validation_metrics.total < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics.total
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics.to_dict(),
            "validation": validation_metrics.to_dict(),
            "elapsed_seconds": time.time() - epoch_start,
        }
        history.append(epoch_record)
        _write_history(history_path, history)
        checkpoint_provenance = {
            **provenance,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "current_metrics": epoch_record,
        }
        save_pretraining_checkpoint(
            last_checkpoint,
            module,
            metadata,
            epoch=epoch,
            global_step=global_step,
            optimizer=optimizer,
            provenance={**checkpoint_provenance, "checkpoint_role": "last"},
        )
        if improved:
            save_pretraining_checkpoint(
                training_checkpoint,
                module,
                metadata,
                epoch=epoch,
                global_step=global_step,
                optimizer=optimizer,
                provenance={**checkpoint_provenance, "checkpoint_role": "best"},
            )
            extract_location_encoder_checkpoint(
                training_checkpoint,
                location_checkpoint,
                map_location="cpu",
            )
            _write_run_metadata(
                metadata_path,
                config=config,
                text_encoder=describe_text_encoder(module.model),
                provenance=checkpoint_provenance,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
            )

        print(
            f"Epoch {epoch}/{config.training.max_epochs}: "
            f"train={train_metrics.total:.5f}, validation={validation_metrics.total:.5f}, "
            f"best={best_validation_loss:.5f} (epoch {best_epoch})"
        )
        patience = config.training.early_stopping_patience
        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without improvement.")
            break

    if not training_checkpoint.is_file() or not location_checkpoint.is_file():
        raise RuntimeError("Pre-training finished without producing best checkpoints.")
    return PretrainingResult(
        training_checkpoint=training_checkpoint,
        location_checkpoint=location_checkpoint,
        last_checkpoint=last_checkpoint,
        history_path=history_path,
        metadata_path=metadata_path,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        global_step=global_step,
    )
