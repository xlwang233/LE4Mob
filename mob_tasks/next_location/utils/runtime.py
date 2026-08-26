import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict

from le4mob import load_location_encoder_checkpoint
from mob_tasks.next_location.baselines.ctle import (
    CTLE,
    CTLEEmbedding,
    MaskedHour,
    MaskedLM,
    PositionalEncoding,
    TemporalEncoding,
)
from mob_tasks.next_location.baselines.static import StaticEmbed
from mob_tasks.next_location.models.mhsa import TransEncoder
from mob_tasks.next_location.utils.dataloader import (
    NextLocationDataset,
    NextLocationEvaluationDataset,
    collate_fn,
    load_pickle,
    split_path,
)
from mob_tasks.next_location.utils.train import get_performance_dict, test, train_net


def load_le4mob_model(ckpt_path, device, coordinate_order=None):
    """Load the public, trainer-independent LE4Mob location encoder."""
    encoder, metadata, _ = load_location_encoder_checkpoint(ckpt_path, map_location=device)
    if coordinate_order is not None and metadata.coordinate_order != coordinate_order:
        raise ValueError(
            f"Checkpoint coordinate order is {metadata.coordinate_order!r}, "
            f"but coordinate_order={coordinate_order!r}."
        )
    return encoder


def load_embed_model(embed_name, save_dir, exp_num=None, map_location="cpu"):
    if embed_name == "vanilla":
        return None

    ckpt_path = os.path.join(save_dir, f"checkpoint_{exp_num}.pt")
    config_path = os.path.join(save_dir, "conf.json")

    # Load the config json file
    with open(config_path) as f:
        config = json.load(f)
    config = edict(config)
    if embed_name == "ctle":
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        if config["ctle_static"]:
            embed_mat = np.load(os.path.join(save_dir, f"embed_mat_{exp_num}.npy"))
            embed_layer = StaticEmbed(embed_mat)
            return embed_layer
        else:
            embed_model, _ = init_embed_models(config)
            embed_model.load_state_dict(ckpt)
            embed_model.eval()
            # embed_model.to(device)
            return embed_model
    else:
        embed_mat = np.load(os.path.join(save_dir, f"embed_mat_{exp_num}.npy"))
        embed_layer = StaticEmbed(embed_mat)
        return embed_layer


def load_config(path):
    """
    Loads config file:
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for _, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_trained_nets(config, model, train_loader, val_loader, test_loader, device, log_dir, logger):
    best_model, performance, training_state = train_net(
        config, model, train_loader, val_loader, test_loader, device, log_dir, logger
    )
    performance["type"] = "vali"

    return best_model, performance, training_state


def get_test_result(config, best_model, test_loader, device, logger):
    return_dict, result_arr_user, df_res = test(config, best_model, test_loader, device, logger)

    performance = get_performance_dict(return_dict)
    performance["type"] = "test"
    # print(performance)

    result_user_df = pd.DataFrame(result_arr_user).T
    result_user_df.columns = [
        "correct@1",
        "correct@3",
        "correct@5",
        "correct@10",
        "rr",
        "ndcg",
        "total",
    ]
    result_user_df.index.name = "user"

    return performance, result_user_df, df_res


def init_embed_models(config, logger=None):
    if config["embed_name"] == "ctle":
        encoding_layer = PositionalEncoding(config.embed_size, max_len=1000)
        if config["encoding_type"] == "temporal":
            encoding_layer = TemporalEncoding(config["embed_size"])
        obj_models = [MaskedLM(config["embed_size"], config["total_loc_num"])]
        if config["ctle_objective"] == "mh":
            obj_models.append(MaskedHour(config["embed_size"]))
        obj_models = nn.ModuleList(obj_models)

        ctle_embedding = CTLEEmbedding(encoding_layer, config["embed_size"], config["total_loc_num"])
        embed_model = CTLE(
            ctle_embedding,
            config["hidden_size"],
            num_layers=config["ctle_num_layers"],
            num_heads=config["ctle_num_heads"],
            init_param=config["init_param"],
            detach=config["ctle_detach"],
        )

        total_params = sum(p.numel() for p in embed_model.parameters() if p.requires_grad)
        if logger is not None:
            logger.info(f"Total number of trainable parameters: {total_params}")
            logger.info(embed_model)

        return embed_model, obj_models
    raise ValueError(f"Unsupported embedding pretraining method: {config['embed_name']}")


def get_models(config, device, logger, exp_num):
    config = edict(config)
    if config.loc_embed_method == "le4mob":
        loc_enc = load_le4mob_model(
            config["loc_enc_ckpt_path"],
            device=device,
            coordinate_order=config.coordinate_order,
        )
    else:
        loc_enc = load_embed_model(
            config["loc_embed_method"], config["loc_enc_ckpt_path"], exp_num=exp_num, map_location=device
        )

    # freeze the loc_enc
    if loc_enc is not None:
        for param in loc_enc.parameters():
            param.requires_grad = False
        embedding_dim = getattr(loc_enc, "output_dim", None)
        embedding_dim = getattr(loc_enc, "embed_size", embedding_dim)
        if embedding_dim is None and hasattr(loc_enc, "embed"):
            embedding_dim = loc_enc.embed.shape[-1]
        if embedding_dim != config.base_emb_size:
            raise ValueError(
                f"Location embedding dimension {embedding_dim} does not match "
                f"base_emb_size={config.base_emb_size}."
            )

    if config.networkName != "mhsa":
        raise ValueError(f"Unsupported next location prediction model: {config.networkName}")
    model = TransEncoder(
        config=config,
        total_loc_num=config.total_loc_num,
        loc_embbedder=loc_enc,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False)

    if logger is not None:
        logger.info(f"Total number of trainable parameters: {total_params}")
        logger.info(model)

    return model


def get_dataloaders(config, logger, exp_num):
    # def get_dataloaders(config):

    dataset_train = NextLocationDataset(
        config.source_root,
        dataset=config.dataset,
        split="train",
        previous_day=config.previous_day,
        inductive_pct=config["inductive_pct"],
        run_id=exp_num,
    )
    dataset_val = NextLocationDataset(
        config.source_root,
        dataset=config.dataset,
        split="validation",
        previous_day=config.previous_day,
        inductive_pct=config["inductive_pct"],
        run_id=exp_num,
    )
    dataset_test = NextLocationEvaluationDataset(
        config.source_root,
        dataset=config.dataset,
        previous_day=config.previous_day,
        inductive_pct=config["inductive_pct"],
        run_id=exp_num,
        stratum="whole_test",
    )

    kwds_train = {
        "shuffle": True,
        "num_workers": config["num_workers"],
        "drop_last": True,
        "batch_size": config["batch_size"],
        "pin_memory": False,
        # "sampler": train_sampler,
    }
    kwds_val = {
        # "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": False,
        # "sampler": val_sampler,
    }
    kwds_test = {
        # "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": False,
        # "sampler": test_sampler,
    }
    fn = collate_fn

    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=fn, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=fn, **kwds_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=fn, **kwds_test)

    if logger is not None:
        logger.info(f"The length of train_loader: {len(train_loader)}")
        logger.info(f"The length of val_loader: {len(val_loader)}")
        logger.info(f"The length of test_loader: {len(test_loader)}")
    else:
        print(f"The length of train_loader: {len(train_loader)}")
        print(f"The length of val_loader: {len(val_loader)}")
        print(f"The length of test_loader: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def get_test_loader(
    config,
    exp_num=0,
    *,
    stratum="whole_test",
    logger=None,
):
    """Build only the requested test loader, without loading train/validation data."""
    dataset = NextLocationEvaluationDataset(
        config.source_root,
        dataset=config.dataset,
        previous_day=config.previous_day,
        inductive_pct=config["inductive_pct"],
        run_id=exp_num,
        stratum=stratum,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        pin_memory=False,
    )
    message = f"The length of the {stratum} loader: {len(loader)}"
    logger.info(message) if logger is not None else print(message)
    return loader


def get_logger(logger_name, log_dir, time_now):
    # Create a logger instance
    logger = logging.getLogger(f"{logger_name}.{os.path.basename(log_dir)}")
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set its log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a file handler and set its log level
    log_file = "log_file" + time_now + ".log"
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def save_args(args, path):
    """
    Save args to json file
    """
    file_path = path + "/" + "args.txt"
    with open(file_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)


def get_training_corpus(data_dir_root, dataset="fsq_nyc", inductive_pct=0, run_id=0):
    """
    Get the training corpus from training data
    """
    train_file_path = split_path(
        data_dir_root,
        dataset,
        "train",
        previous_day=7,
        inductive_pct=inductive_pct,
        run_id=run_id,
    )
    train_data = load_pickle(train_file_path)
    train_sentences = []
    train_users = []
    train_weekdays = []
    train_start_mins = []
    # train_corpus is a list of list of numbers
    for iter_dict in train_data:
        X = iter_dict["X"].tolist()
        Y = iter_dict["Y"]
        X.append(Y)
        train_sentences.append(X)
        train_users.append(iter_dict["user_X"])
        weekday_X = iter_dict["weekday_X"].tolist()
        weekday_Y = iter_dict["weekday_Y"]
        weekday_X.append(weekday_Y)
        train_weekdays.append(weekday_X)
        start_min_X = iter_dict["start_min_X"].tolist()
        start_min_Y = iter_dict["start_min_Y"]
        start_min_X.append(start_min_Y)
        train_start_mins.append(start_min_X)
    return train_sentences, train_users, train_weekdays, train_start_mins


def get_id2coor_df(data_dir_root, dataset="fsq_nyc"):
    loc_id_file = os.path.join(data_dir_root, dataset, f"{dataset}_locs.csv")
    if os.path.isfile(loc_id_file):
        id2coor_df = pd.read_csv(loc_id_file)
    else:
        dataset_root = os.path.join(data_dir_root, dataset)
        mapping_path = os.path.join(dataset_root, f"{dataset}_location_id.csv")
        locations_path = os.path.join(dataset_root, f"locations_{dataset}.csv")
        mapping = pd.read_csv(mapping_path)
        locations = pd.read_csv(locations_path)
        if "location_id" not in locations.columns and "id" in locations.columns:
            locations.rename(columns={"id": "location_id"}, inplace=True)
        if not {"longitude", "latitude"}.issubset(locations.columns) and "center" in locations:
            extracted = locations.center.astype("string").str.extract(
                r"^\s*POINT\s*\(\s*([^\s]+)\s+([^\s]+)\s*\)\s*$",
                expand=True,
            )
            if extracted.isna().any(axis=None):
                raise ValueError(f"{locations_path} contains an invalid WKT POINT center.")
            locations["longitude"] = pd.to_numeric(extracted[0], errors="raise")
            locations["latitude"] = pd.to_numeric(extracted[1], errors="raise")
        id2coor_df = mapping.merge(
            locations[["location_id", "longitude", "latitude"]],
            on="location_id",
            how="left",
            validate="one_to_one",
        )
    # drop the location_id column and set the "location_id_num" as the index
    if "location_id" in id2coor_df.columns:
        id2coor_df.drop(columns=["location_id"], inplace=True)
    id2coor_df.set_index("location_id_num", inplace=True)
    # if the df contains columns "latitude" and "longitude", then we need to rename them
    if "latitude" in id2coor_df.columns:
        id2coor_df.rename(columns={"latitude": "lat", "longitude": "lng"}, inplace=True)
    return id2coor_df


def get_tale_training_corpus(data_dir_root, dataset="fsq_nyc", inductive_pct=0, run_id=0):
    train_file_path = split_path(
        data_dir_root,
        dataset,
        "train",
        previous_day=7,
        inductive_pct=inductive_pct,
        run_id=run_id,
    )
    train_data = load_pickle(train_file_path)
    train_sentences = []
    train_timestamps = []
    # train_corpus is a list of list of numbers
    for iter_dict in train_data:
        X = iter_dict["X"].tolist()
        Y = iter_dict["Y"]
        X.append(Y)
        train_sentences.append(X)
        timestamp_X = iter_dict["timestamp_X"].tolist()
        timestamp_Y = iter_dict["timestamp_Y"]
        timestamp_X.append(timestamp_Y)
        train_timestamps.append(timestamp_X)
    return train_sentences, train_timestamps
