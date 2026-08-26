"""Load the prepared next-location splits."""

from __future__ import annotations

import csv
import gzip
import pickle
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence


INDUCTIVE_EVALUATION_STRATA = ("combined_inductive",)
EVALUATION_STRATA = ("whole_test", *INDUCTIVE_EVALUATION_STRATA)


def split_path(
    source_root: str | Path,
    dataset: str,
    split: str,
    *,
    previous_day: int,
    inductive_pct: int,
    run_id: int,
) -> Path:
    """Resolve one prepared split using the repository naming convention."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    if inductive_pct > 0 and split != "test":
        name = f"{dataset}_{previous_day}_{split}_remove{inductive_pct}pct_{run_id}.pk"
    else:
        name = f"{dataset}_{previous_day}_{split}.pk"
    path = Path(source_root) / dataset / name
    compressed_path = Path(f"{path}.gz")
    return compressed_path if compressed_path.is_file() else path


def evaluation_path(
    source_root: str | Path,
    dataset: str,
    *,
    inductive_pct: int,
    run_id: int,
    stratum: str,
) -> Path:
    """Resolve the authoritative membership file for one evaluation set."""
    if stratum not in EVALUATION_STRATA:
        raise ValueError(f"Unknown evaluation stratum: {stratum!r}.")
    dataset_root = Path(source_root) / dataset
    if stratum == "whole_test":
        return dataset_root / "conventional_test_set" / "whole_test.csv"
    if inductive_pct == 0:
        raise ValueError("Inductive evaluation requires inductive_pct > 0.")
    split_root = dataset_root / "stratified_inductive_test_sets" / f"split_{run_id}"
    return split_root / f"{stratum}.csv"


def load_pickle(path: str | Path):
    """Load either a gzip-compressed or an uncompressed prepared pickle."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    with path.open("rb") as handle:
        return pickle.load(handle)


class NextLocationDataset(torch.utils.data.Dataset):
    """A prepared sequence split for next-location prediction."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        dataset: str,
        split: str,
        previous_day: int = 7,
        inductive_pct: int = 0,
        run_id: int = 0,
    ) -> None:
        path = split_path(
            source_root,
            dataset,
            split,
            previous_day=previous_day,
            inductive_pct=inductive_pct,
            run_id=run_id,
        )
        if not path.is_file():
            raise FileNotFoundError(f"Prepared next-location split not found: {path}")
        self.records = load_pickle(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        source = torch.as_tensor(record["X"], dtype=torch.long)
        target = torch.as_tensor(record["Y"], dtype=torch.long)
        context = {
            "user": torch.as_tensor(record["user_X"], dtype=torch.long),
            "time": torch.as_tensor(record["start_min_X"] // 30, dtype=torch.long),
            "start_min_X": torch.as_tensor(record["start_min_X"], dtype=torch.long),
            "start_min_Y": torch.as_tensor(record["start_min_Y"], dtype=torch.long),
            "diff": torch.as_tensor(record["diff"]),
            "weekday_X": torch.as_tensor(record["weekday_X"], dtype=torch.long),
            "weekday_Y": torch.as_tensor(record["weekday_Y"], dtype=torch.long),
            "lat_X": torch.as_tensor(record["lat_X"], dtype=torch.float64),
            "lon_X": torch.as_tensor(record["lon_X"], dtype=torch.float64),
            "x_X": torch.as_tensor(record["x_X"], dtype=torch.float64),
            "y_X": torch.as_tensor(record["y_X"], dtype=torch.float64),
        }
        return source, target, context


class NextLocationEvaluationDataset(NextLocationDataset):
    """Select evaluation samples using the exported, protocol-defined test indices."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        dataset: str,
        previous_day: int = 7,
        inductive_pct: int = 0,
        run_id: int = 0,
        stratum: str = "whole_test",
    ) -> None:
        source_path = split_path(
            source_root,
            dataset,
            "test",
            previous_day=previous_day,
            inductive_pct=0,
            run_id=run_id,
        )
        membership_path = evaluation_path(
            source_root,
            dataset,
            inductive_pct=inductive_pct,
            run_id=run_id,
            stratum=stratum,
        )
        if not source_path.is_file():
            raise FileNotFoundError(f"Prepared next-location test split not found: {source_path}")
        if not membership_path.is_file():
            raise FileNotFoundError(f"Evaluation membership not found: {membership_path}")

        source_records = load_pickle(source_path)
        with membership_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "test_index" not in rows[0] or "target_location_id" not in rows[0]:
            raise ValueError(f"Invalid evaluation membership schema: {membership_path}")

        indices = [int(row["test_index"]) for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate test_index values in {membership_path}")
        if min(indices) < 0 or max(indices) >= len(source_records):
            raise ValueError(f"Out-of-range test_index in {membership_path}")

        self.records = [source_records[index] for index in indices]
        self.test_indices = indices
        for row, record in zip(rows, self.records):
            target = int(record["Y"])
            if target == 1 or target != int(row["target_location_id"]):
                raise ValueError(
                    f"Evaluation membership target mismatch at test_index={row['test_index']}."
                )

    def __getitem__(self, index: int):
        source, target, context = super().__getitem__(index)
        context["test_index"] = torch.tensor(self.test_indices[index], dtype=torch.long)
        return source, target, context


def collate_fn(batch):
    """Pad variable-length histories into sequence-first batches."""
    sources, targets, contexts = zip(*batch)
    context_batch = {"len": torch.tensor([len(source) for source in sources])}
    for key in contexts[0]:
        values = [context[key] for context in contexts]
        if key in {"user", "start_min_Y", "weekday_Y", "test_index"}:
            context_batch[key] = torch.stack(values)
        else:
            context_batch[key] = pad_sequence(values)
    return pad_sequence(sources), torch.stack(targets), context_batch
