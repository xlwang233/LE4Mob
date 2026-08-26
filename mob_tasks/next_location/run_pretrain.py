import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from easydict import EasyDict as edict

from mob_tasks.next_location.baselines.ctle import train_ctle
from mob_tasks.next_location.baselines.poi2vec import P2VData, POI2Vec
from mob_tasks.next_location.baselines.tale import Tale, TaleData, train_tale
from mob_tasks.next_location.baselines.teaser import Teaser, TeaserData, train_teaser
from mob_tasks.next_location.baselines.w2v import SkipGram, SkipGramData, train_skipgram
from mob_tasks.next_location.utils.runtime import (
    get_dataloaders,
    get_id2coor_df,
    get_logger,
    get_tale_training_corpus,
    get_training_corpus,
    init_embed_models,
    load_config,
    setup_seed,
)


def init_save_path(config, time_now):
    """define the path to save, and save the configuration file."""
    if config["inductive_pct"] > 0:
        embed_name = f"{config.dataset}_{config.embed_name}_inductive{config.inductive_pct}pct"
    else:
        embed_name = f"{config.dataset}_{config.embed_name}"
    log_dir = os.path.join(config.save_root, f"{embed_name}_{time_now}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = get_logger("my_logger", log_dir, time_now)

    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    return logger, log_dir


def main(config, time_now):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    # train_loader, val_loader, _ = get_dataloaders(config=config)
    # save the conf
    logger, log_dir = init_save_path(config, time_now)

    if config["inductive_pct"] > 0:
        num_runs = config.get("num_runs", 5)
    else:
        num_runs = config.get("num_runs", 1)

    if config.embed_name == "ctle":
        for i in range(num_runs):
            setup_seed(config.get("seed", 42) + i)
            train_loader, val_loader, _ = get_dataloaders(config=config, logger=logger, exp_num=i)
            # get model
            embed_model, obj_models = init_embed_models(config, logger)
            embed_layer = train_ctle(
                train_loader,
                val_loader,
                embed_model,
                obj_models,
                mask_prop=config.ctle_mask_prop,
                num_epoch=config.num_epoch,
                device=device,
                logger=logger,
                log_dir=log_dir,
                config=config,
                run_id=i,
            )

            if config.save_static:
                embed_mat = embed_layer.static_embed()
                # embed_layer = StaticEmbed(embed_mat)
                # save the embed_mat
                np.save(os.path.join(log_dir, f"embed_mat_{i}.npy"), embed_mat)

    elif config.embed_name in ["skipgram", "cbow"]:
        for i in range(num_runs):
            setup_seed(config.get("seed", 42) + i)
            embed_train_sentences, _, _, _ = get_training_corpus(
                data_dir_root=config["source_root"],
                dataset=config["dataset"],
                inductive_pct=config["inductive_pct"],
                run_id=i,
            )  # list of list of int
            sg_dataset = SkipGramData(embed_train_sentences)
            sg_model = SkipGram(
                config["total_loc_num"], config["embed_size"], cbow=(config["embed_name"] == "cbow")
            )
            embed_mat = train_skipgram(
                sg_model,
                sg_dataset,
                window_size=config["w2v_window_size"],
                num_neg=config["skipgram_neg"],
                batch_size=config["batch_size"],
                num_epoch=config["num_epoch"],
                init_lr=config["learning_rate"],
                device=device,
                logger=logger,
            )
            np.save(os.path.join(log_dir, f"embed_mat_{i}.npy"), embed_mat)
    elif config.embed_name == "tale":
        for i in range(num_runs):
            setup_seed(config.get("seed", 42) + i)
            # logger.info(f"Begin the {i}th run of the pre-train experiment.")
            embed_train_sentences, embed_train_timestamp = get_tale_training_corpus(
                data_dir_root=config["source_root"],
                dataset=config["dataset"],
                inductive_pct=config["inductive_pct"],
                run_id=i,
            )  # list of list of int
            tale_dataset = TaleData(
                embed_train_sentences,
                embed_train_timestamp,
                config["tale_slice"],
                config["tale_span"],
                indi_context=config["tale_indi_context"],
            )
            tale_model = Tale(config["total_loc_num"], len(tale_dataset.id2index), config["embed_size"])
            embed_mat = train_tale(
                tale_model,
                tale_dataset,
                config["w2v_window_size"],
                batch_size=config["batch_size"],
                num_epoch=config["num_epoch"],
                init_lr=config["learning_rate"],
                device=device,
                logger=logger,
            )
            np.save(os.path.join(log_dir, f"embed_mat_{i}.npy"), embed_mat)

    elif config.embed_name == "poi2vec":
        for i in range(num_runs):
            setup_seed(config.get("seed", 42) + i)
            embed_train_sentences, _, _, _ = get_training_corpus(
                data_dir_root=config["source_root"],
                dataset=config["dataset"],
                inductive_pct=config["inductive_pct"],
                run_id=i,
            )  # list of list of int
            id2coor_df = get_id2coor_df(data_dir_root=config["source_root"], dataset=config["dataset"])
            poi2vec_data = P2VData(
                embed_train_sentences,
                id2coor_df,
                theta=config["poi2vec_theta"],
                indi_context=config["poi2vec_indi_context"],
            )
            poi2vec_model = POI2Vec(config["total_loc_num"], poi2vec_data.total_offset, config["embed_size"])
            embed_mat = train_tale(
                poi2vec_model,
                poi2vec_data,
                config["w2v_window_size"],
                batch_size=config["batch_size"],
                num_epoch=config["num_epoch"],
                init_lr=config["learning_rate"],
                device=device,
                logger=logger,
            )
            np.save(os.path.join(log_dir, f"embed_mat_{i}.npy"), embed_mat)
    elif config.embed_name == "teaser":
        for i in range(num_runs):
            setup_seed(config.get("seed", 42) + i)
            embed_train_sentences, embed_train_users, embed_train_weekdays, _ = get_training_corpus(
                data_dir_root=config["source_root"],
                dataset=config["dataset"],
                inductive_pct=config["inductive_pct"],
                run_id=i,
            )
            coor_mat = get_id2coor_df(data_dir_root=config["source_root"], dataset=config["dataset"])
            # Pandas nullable Float64 columns otherwise produce an object array, which
            # NumPy distance ufuncs cannot process reliably.
            coor_mat = coor_mat.reset_index().to_numpy(dtype=np.float64)
            teaser_dataset = TeaserData(
                embed_train_users,
                embed_train_sentences,
                embed_train_weekdays,
                coor_mat,
                num_ne=config["teaser_num_ne"],
                num_nn=config["teaser_num_nn"],
                indi_context=config["teaser_indi_context"],
            )
            # print("111")
            teaser_model = Teaser(
                num_vocab=config["total_loc_num"],
                num_user=config["total_user_num"],
                embed_dimension=config["embed_size"],
                week_embed_dimension=config["teaser_week_embed_size"],
                beta=config["teaser_beta"],
            )
            embed_mat = train_teaser(
                teaser_model,
                teaser_dataset,
                window_size=config["w2v_window_size"],
                num_neg=config["skipgram_neg"],
                batch_size=config["batch_size"],
                num_epoch=config["num_epoch"],
                init_lr=config["learning_rate"],
                device=device,
                logger=logger,
            )
            np.save(os.path.join(log_dir, f"embed_mat_{i}.npy"), embed_mat)
    else:
        raise ValueError(f"Unsupported baseline: {config.embed_name}")

    print("Pre-training finished!")


def build_parser():
    parser = argparse.ArgumentParser(description="Pre-train a mobility-derived location baseline.")
    parser.add_argument("--config", required=True, help="Baseline pretraining config file path.")
    parser.add_argument(
        "--inductive-pct",
        type=int,
        choices=(0, 10),
        required=True,
        help="Experimental setting: 0 for conventional or 10 for the provided inductive splits.",
    )
    return parser


def load_run_config(config_path, inductive_pct):
    """Load a method config and attach the setting selected explicitly by the CLI."""
    config = load_config(config_path)
    config["inductive_pct"] = inductive_pct
    return edict(config)


def cli_main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_run_config(args.config, args.inductive_pct)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    main(config, time_now=formatted_datetime)


if __name__ == "__main__":
    cli_main()
