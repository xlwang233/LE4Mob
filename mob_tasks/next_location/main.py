"""Train and evaluate the MHSA model for next location prediction."""

from __future__ import annotations

import argparse
import json
import numbers
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from easydict import EasyDict as edict

from le4mob.checkpoint import sha256_file
from mob_tasks.checkpoint import (
    DownstreamCheckpointMetadata,
    save_downstream_checkpoint,
)
from mob_tasks.next_location.utils.dataloader import INDUCTIVE_EVALUATION_STRATA
from mob_tasks.next_location.utils.runtime import (
    get_dataloaders,
    get_logger,
    get_models,
    get_test_loader,
    get_test_result,
    get_trained_nets,
    load_config,
    setup_seed,
)


def init_save_path(config, time_now: str, run_id: int):
    """Create the run directory and persist its resolved configuration."""
    experiment_name = f"{config.dataset}_{config.loc_embed_method}_{config.networkName}"
    output_root = config.inductive_save_root if config.inductive_pct > 0 else config.save_root
    log_dir = Path(output_root) / f"{experiment_name}_{time_now}_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("next_location", str(log_dir), time_now)
    (log_dir / "conf.json").write_text(json.dumps(dict(config), indent=2, sort_keys=True))
    setting = "inductive" if config.inductive_pct > 0 else "conventional"
    logger.info(f"Begin run {run_id} in the {setting} setting.")
    return logger, log_dir


def _float_metrics(performance):
    return {
        key: float(value)
        for key, value in performance.items()
        if key != "type" and isinstance(value, numbers.Real)
    }


def _location_checkpoint_provenance(config):
    if config.loc_embed_method != "le4mob":
        return None, None
    checkpoint = Path(config.loc_enc_ckpt_path)
    return str(checkpoint), sha256_file(checkpoint)


def save_reproducibility_checkpoint(
    model,
    config,
    performance,
    training_state,
    seed: int,
    run_id: int,
    log_dir: Path,
):
    """Save a versioned downstream checkpoint with embedding provenance."""
    location_checkpoint, location_checksum = _location_checkpoint_provenance(config)
    split = "conventional" if config.inductive_pct == 0 else f"inductive_{config.inductive_pct}pct"
    metadata = DownstreamCheckpointMetadata(
        task="next_location",
        model_name=config.networkName,
        model_config=dict(config),
        dataset=config.dataset,
        data_split=split,
        seed=seed,
        location_embedding_name=config.loc_embed_method,
        location_checkpoint=location_checkpoint,
        location_checkpoint_sha256=location_checksum,
        metrics=_float_metrics(performance),
        extra={"run_id": run_id},
    )
    save_downstream_checkpoint(
        log_dir / "downstream_checkpoint.pt",
        model,
        metadata,
        epoch=training_state["epoch"],
        global_step=training_state["global_step"],
        external_state_prefixes=("Embedding.emb_loc.",)
        if config.loc_embed_method == "le4mob"
        else (),
    )


def resolve_run_seed(config, run_id: int) -> int:
    """Resolve either legacy per-run seeds or a fixed campaign seed."""
    seed = config.get("seed", 42)
    if config.get("vary_seed_by_run", True):
        seed += run_id
    return int(seed)


def single_run(config, device, time_now: str, run_id: int):
    seed = resolve_run_seed(config, run_id)
    setup_seed(seed)
    logger, log_dir = init_save_path(config, time_now, run_id)
    train_loader, val_loader, test_loader = get_dataloaders(config, logger, run_id)
    embedding_run = run_id if config.inductive_pct > 0 else 0
    model = get_models(config, device, logger, exp_num=embedding_run)

    model, validation_performance, training_state = get_trained_nets(
        config, model, train_loader, val_loader, test_loader, device, str(log_dir), logger
    )
    test_performance, user_performance, predictions = get_test_result(
        config, model, test_loader, device, logger
    )
    test_performance["type"] = "whole_test"
    user_performance.to_csv(log_dir / "user_detail_whole_test.csv")
    predictions.to_csv(log_dir / "predictions_whole_test.csv", index=False)

    results = [validation_performance, test_performance]
    checkpoint_metrics = test_performance
    if config.inductive_pct > 0:
        for stratum in INDUCTIVE_EVALUATION_STRATA:
            loader = get_test_loader(
                config,
                exp_num=run_id,
                stratum=stratum,
                logger=logger,
            )
            performance, user_detail, stratum_predictions = get_test_result(
                config, model, loader, device, logger
            )
            performance["type"] = stratum
            user_detail.to_csv(log_dir / f"user_detail_{stratum}.csv")
            stratum_predictions.to_csv(
                log_dir / f"predictions_{stratum}.csv",
                index=False,
            )
            results.append(performance)
            if stratum == "combined_inductive":
                checkpoint_metrics = performance

    save_reproducibility_checkpoint(
        model,
        config,
        checkpoint_metrics,
        training_state,
        seed,
        run_id,
        log_dir,
    )
    logger.info(f"Saved downstream checkpoint to {log_dir / 'downstream_checkpoint.pt'}")

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    return results, log_dir


def main(config, time_now: str):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    all_results = []
    last_log_dir = None
    for run_id in range(config.get("num_runs", 5)):
        results, last_log_dir = single_run(config, device, time_now, run_id)
        all_results.extend(results)

    summary_path = Path(last_log_dir).parent / f"{config.dataset}_{config.networkName}_{time_now}.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    print(f"Finished next location prediction. Results: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Next location prediction config file path.")
    args = parser.parse_args()
    resolved_config = edict(load_config(args.config))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main(resolved_config, timestamp)
