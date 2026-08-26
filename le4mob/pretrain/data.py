"""CSV data loading and tokenization for LE4Mob pre-training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from le4mob.pretrain.config import PretrainingDataConfig


class POITableDataset(Dataset[dict[str, Any]]):
    """One aligned location/text pair per row of a curated POI table."""

    def __init__(self, config: PretrainingDataConfig) -> None:
        self.config = config
        path = Path(config.path)
        if not path.is_file():
            raise FileNotFoundError(f"POI table not found: {path}")
        frame = pd.read_csv(path)
        required = {
            config.id_column,
            config.text_column,
            *config.coordinate_columns,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"POI table is missing required columns: {missing}")
        if len(frame) < 2:
            raise ValueError("POI table must contain at least two rows.")
        selected = frame[[config.id_column, config.text_column, *config.coordinate_columns]].copy()
        if selected.isna().any().any():
            columns = selected.columns[selected.isna().any()].tolist()
            raise ValueError(f"POI table contains missing values in columns: {columns}")

        coordinates = selected[list(config.coordinate_columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(coordinates).all():
            raise ValueError("POI coordinates must all be finite.")
        self.identifiers = selected[config.id_column].astype(str).tolist()
        self.texts = selected[config.text_column].astype(str).tolist()
        if any(not text.strip() for text in self.texts):
            raise ValueError("POI descriptions cannot be empty.")
        self.coordinates = torch.from_numpy(coordinates)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "identifier": self.identifiers[index],
            "text": self.texts[index],
            "coordinates": self.coordinates[index],
        }


class LocationTextCollator:
    def __init__(self, tokenizer: Any, max_token_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        tokenization_options: dict[str, Any] = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self.max_token_length is not None:
            tokenization_options["max_length"] = self.max_token_length
        encoded = self.tokenizer([row["text"] for row in rows], **tokenization_options)
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise ValueError("Tokenizer must return input_ids and attention_mask.")
        return {
            "identifiers": [row["identifier"] for row in rows],
            "coordinates": torch.stack([row["coordinates"] for row in rows]),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


@dataclass(frozen=True)
class PretrainingDataLoaders:
    dataset: POITableDataset
    train: DataLoader
    validation: DataLoader
    train_sampler: EpochRandomSampler
    train_size: int
    validation_size: int


class EpochRandomSampler(Sampler[int]):
    """Deterministic epoch-specific ordering that also supports exact resume."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        if epoch <= 0:
            raise ValueError("epoch must be positive.")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_pretraining_dataloaders(
    config: PretrainingDataConfig,
    *,
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    training_seed: int,
    pin_memory: bool = False,
) -> PretrainingDataLoaders:
    dataset = POITableDataset(config)
    validation_size = max(1, round(len(dataset) * config.validation_fraction))
    validation_size = min(validation_size, len(dataset) - 1)
    permutation = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(config.split_seed)
    ).tolist()
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]
    collator = LocationTextCollator(tokenizer, config.max_token_length)
    train_subset = Subset(dataset, train_indices)
    train_sampler = EpochRandomSampler(len(train_subset), training_seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_subset,
        sampler=train_sampler,
        **common,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        shuffle=False,
        **common,
    )
    return PretrainingDataLoaders(
        dataset=dataset,
        train=train_loader,
        validation=validation_loader,
        train_sampler=train_sampler,
        train_size=len(train_indices),
        validation_size=len(validation_indices),
    )
