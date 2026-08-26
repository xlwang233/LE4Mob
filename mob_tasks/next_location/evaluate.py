"""Evaluate a saved next location model without retraining it."""

from __future__ import annotations

import argparse
import json

from mob_tasks.next_location.evaluation import evaluate_downstream_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Versioned downstream checkpoint.")
    parser.add_argument("--output-directory", help="Directory for metrics and prediction CSVs.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--data-root", help="Override the data root stored in checkpoint metadata.")
    parser.add_argument(
        "--location-checkpoint",
        help="Override the external encoder path; its checksum must match checkpoint metadata.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_downstream_checkpoint(
        args.checkpoint,
        output_directory=args.output_directory,
        device=args.device,
        data_root=args.data_root,
        location_checkpoint=args.location_checkpoint,
    )
    print(
        json.dumps(
            {name: evaluation.metrics for name, evaluation in result.evaluations.items()},
            indent=2,
        )
    )
    if result.output_directory is not None:
        print(f"Evaluation artifacts: {result.output_directory}")


if __name__ == "__main__":
    main()
