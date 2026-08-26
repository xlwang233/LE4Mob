"""Run LE4Mob pre-training without a trainer framework."""

from __future__ import annotations

import argparse

from le4mob.pretrain.config import PretrainingRunConfig
from le4mob.pretrain.trainer import run_pretraining


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pre-training YAML config file.")
    parser.add_argument(
        "--setting",
        required=True,
        choices=("conventional", "inductive"),
        help="Pre-training setting profile to resolve from the config.",
    )
    parser.add_argument(
        "--data-path",
        help="Override data.path, useful when curated POI tables live outside the repository.",
    )
    parser.add_argument("--output-directory", help="Override output.directory.")
    parser.add_argument("--device", help="Override training.device, for example cuda:0 or cpu.")
    parser.add_argument("--resume", help="Compact LE4Mob training checkpoint from which to resume.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PretrainingRunConfig.from_yaml(args.config, setting=args.setting)
    result = run_pretraining(
        config,
        data_path=args.data_path,
        output_directory=args.output_directory,
        device=args.device,
        resume_from=args.resume,
    )
    print(f"Best compact training checkpoint: {result.training_checkpoint}")
    print(f"Lightweight location checkpoint: {result.location_checkpoint}")
    print(f"Run metadata: {result.metadata_path}")
    print(f"Best validation loss: {result.best_validation_loss:.6f} at epoch {result.best_epoch}")


if __name__ == "__main__":
    main()
