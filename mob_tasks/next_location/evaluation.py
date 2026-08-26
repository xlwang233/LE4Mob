"""Reusable test-only evaluation for versioned next location checkpoints."""

from __future__ import annotations

import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from easydict import EasyDict as edict

from le4mob.checkpoint import sha256_file
from mob_tasks.checkpoint import (
    load_downstream_checkpoint,
    read_downstream_checkpoint_metadata,
)
from mob_tasks.next_location.utils.dataloader import INDUCTIVE_EVALUATION_STRATA
from mob_tasks.next_location.utils.runtime import get_models, get_test_loader, get_test_result


@dataclass(frozen=True)
class TestEvaluation:
    metrics: dict[str, float | str]
    user_performance: pd.DataFrame
    predictions: pd.DataFrame


@dataclass(frozen=True)
class DownstreamEvaluationResult:
    evaluations: dict[str, TestEvaluation]
    output_directory: Path | None

    @property
    def ordinary(self) -> TestEvaluation:
        return self.evaluations["whole_test"]

    @property
    def inductive(self) -> TestEvaluation | None:
        return self.evaluations.get("combined_inductive")


def _json_metrics(metrics: dict[str, Any]) -> dict[str, float | str]:
    return {
        key: float(value) if isinstance(value, numbers.Real) else str(value) for key, value in metrics.items()
    }


def _evaluate_loader(config, model, loader, device) -> TestEvaluation:
    metrics, user_performance, predictions = get_test_result(config, model, loader, device, logger=None)
    return TestEvaluation(
        metrics=_json_metrics(metrics),
        user_performance=user_performance,
        predictions=predictions,
    )


def _write_evaluation(result: DownstreamEvaluationResult, metadata: dict[str, Any]) -> None:
    if result.output_directory is None:
        return
    root = result.output_directory
    root.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"checkpoint_metadata": metadata}
    for stratum, evaluation in result.evaluations.items():
        evaluation.user_performance.to_csv(root / f"user_detail_{stratum}.csv")
        evaluation.predictions.to_csv(root / f"predictions_{stratum}.csv", index=False)
        metrics[stratum] = evaluation.metrics
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def evaluate_downstream_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    device: str = "auto",
    data_root: str | Path | None = None,
    location_checkpoint: str | Path | None = None,
) -> DownstreamEvaluationResult:
    """Load one versioned checkpoint and evaluate its ordinary and optional inductive tests."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    header = read_downstream_checkpoint_metadata(checkpoint_path)
    metadata = header.metadata
    if metadata.task != "next_location":
        raise ValueError(f"Expected a next location checkpoint, got task={metadata.task!r}.")

    config = edict(dict(metadata.model_config))
    config.verbose = False
    if data_root is not None:
        config.source_root = str(Path(data_root).expanduser())
    run_index = int(metadata.extra.get("run_index", metadata.extra.get("run_id", 0)))
    expected_location_checksum = metadata.location_checkpoint_sha256
    if expected_location_checksum is not None:
        configured_location = location_checkpoint or metadata.location_checkpoint
        if configured_location is None:
            raise ValueError("Checkpoint metadata is missing its external location checkpoint path.")
        configured_location = Path(configured_location).expanduser()
        if not configured_location.is_file():
            raise FileNotFoundError(
                f"External location checkpoint not found: {configured_location}. "
                "Use --location-checkpoint to provide the same checksummed artifact."
            )
        actual_checksum = sha256_file(configured_location)
        if actual_checksum != expected_location_checksum:
            raise ValueError(
                "External location checkpoint checksum mismatch: "
                f"expected {expected_location_checksum}, got {actual_checksum}."
            )
        config.loc_enc_ckpt_path = str(configured_location)

    resolved_device = torch.device(
        "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    embedding_run = run_index if config.inductive_pct > 0 else 0
    model = get_models(config, resolved_device, logger=None, exp_num=embedding_run)
    load_downstream_checkpoint(checkpoint_path, model, map_location=resolved_device)
    model.eval()

    whole_loader = get_test_loader(config, exp_num=run_index, logger=None)
    evaluations = {
        "whole_test": _evaluate_loader(config, model, whole_loader, resolved_device)
    }
    evaluations["whole_test"].metrics["type"] = "whole_test"
    if config.inductive_pct > 0:
        for stratum in INDUCTIVE_EVALUATION_STRATA:
            loader = get_test_loader(
                config,
                exp_num=run_index,
                stratum=stratum,
                logger=None,
            )
            evaluations[stratum] = _evaluate_loader(
                config,
                model,
                loader,
                resolved_device,
            )
            evaluations[stratum].metrics["type"] = stratum

    result = DownstreamEvaluationResult(
        evaluations=evaluations,
        output_directory=Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else None,
    )
    _write_evaluation(result, metadata.to_dict())
    return result
