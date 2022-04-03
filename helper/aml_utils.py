import torch
import torch.nn as nn
import os

from helper.subprocess_utils import active_sampling
from helper.path_extractor import search_best_model_path

from yacs.config import CfgNode
import shutil
from copy import deepcopy
import distutils.dir_util

from pycls.core.config import custom_dump_cfg, load_custom_cfg
from typing import Tuple


def automl_model_inference(
    args, cfg: CfgNode, automl_path: str, al_iteration: int = 1
) -> Tuple[str, str]:
    """This function is responsible to produce active sets and mark the best model among all the trial results
    in automl. We mark best by copying the best model into another directory named as best_automl_results.

    Args:
        args: Input args.
        cfg (CfgNode): Reference to config yaml object.
        automl_path (str): Path to all the checkpoints created during AutoML trials.
        al_iteration (int, optional): Current AL iteration. Defaults to 1.

    Returns:
        Tuple[str, str]: Path to results, Path to checkpoint used in AL sampling.
    """

    # Defining model depth
    model_type = cfg.MODEL.TYPE
    model_depth = cfg.MODEL.DEPTH
    temp_dir_specific = cfg.DIR_SPECIFIC

    # Current data partition
    model_select_partition = (
        args.init_partition + (al_iteration - 1) * args.step_partition
    )
    curr_data_partition = args.init_partition + (al_iteration) * args.step_partition

    best_model_path = search_best_model_path(automl_path)
    # copy the best model dir tree from automl trial models to best_auto_ml folder
    _, sub_path = best_model_path.split("auto_ml_results")
    sub_par_path, sub_ckpt_path = sub_path.split("trial-")
    slash_index = sub_ckpt_path.find("/")
    sub_ckpt_path = sub_ckpt_path[slash_index + 1 :]
    dest_path = os.path.join(
        args.out_dir, "best_automl_results", sub_par_path.lstrip("/")
    )
    srcPath = best_model_path.split("checkpoints")[0]

    distutils.dir_util.copy_tree(srcPath, dest_path)
    temp_cfg = deepcopy(cfg)
    # load
    load_custom_cfg(temp_cfg, dest_path)
    # modify
    temp_cfg.OUT_DIR = dest_path
    # save
    temp_cfg.NUM_GPUS = args.n_GPU
    temp_cfg.OPTIM.MAX_EPOCH = args.clf_epochs

    custom_dump_cfg(temp_cfg)
    print("after dumping")
    print("dest_path: ", dest_path)
    cfg.merge_from_file(os.path.join(dest_path, "config.yaml"))

    # Update necessary fields
    cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR = os.path.join(
        cfg.OUT_DIR, "checkpoints", os.path.basename(best_model_path)
    )  # best_model_path
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.sampling_fn
    cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS = args.dropout_iterations
    cfg.NUM_GPUS = args.n_GPU
    cfg.PORT = args.port
    cfg.DIR_SPECIFIC = temp_dir_specific
    cfg.MODEL.DEPTH = model_depth
    cfg.ACTIVE_LEARNING.DATA_SPLIT = curr_data_partition
    cfg.OPTIM.MAX_EPOCH = args.clf_epochs
    cfg.LOG_PERIOD = args.log_iter

    cfg.SWA_MODE.ACTIVATE = args.swa_mode
    cfg.SWA_MODE.FREQ = args.swa_freq
    cfg.SWA_MODE.LR = args.swa_lr
    cfg.SWA_MODE.START_ITER = args.swa_iter

    cfg.VAAL.TRAIN_VAAL = True if args.sampling_fn.lower() == "vaal" else False
    cfg.VAAL.Z_DIM = args.vaal_z_dim
    cfg.VAAL.VAE_BS = args.vaal_vae_bs
    cfg.VAAL.VAE_EPOCHS = args.vaal_epochs
    cfg.VAAL.VAE_LR = args.vaal_vae_lr
    cfg.VAAL.DISC_LR = args.vaal_disc_lr
    cfg.VAAL.BETA = args.vaal_beta
    cfg.VAAL.ADVERSARY_PARAM = args.vaal_adv_param

    # NOISY EXPS
    cfg.ACTIVE_LEARNING.NOISY_ORACLE = args.noisy_oracle

    if args.isTransferExp:
        cfg.TRAIN.TRANSFER_EXP = args.isTransferExp
        cfg.MODEL.TRANSFER_MODEL_TYPE = args.transfer_model_type
        cfg.MODEL.TRANSFER_MODEL_DEPTH = args.transfer_model_depth
        cfg.MODEL.TRANSFER_MODEL_STYLE = args.transfer_model_style
        cfg.MODEL.TRANSFER_DIR_SPECIFIC = args.transfer_dir_specific

    temp_prev_path, temp_next_path = cfg.OUT_DIR.split(str(model_select_partition))
    if al_iteration >= 2:
        cfg.OUT_DIR = os.path.join(
            temp_prev_path, str(curr_data_partition), temp_next_path.lstrip("/")
        )
    else:
        cfg.OUT_DIR = os.path.join(
            temp_prev_path,
            str(curr_data_partition),
            args.sampling_fn,
            temp_next_path.lstrip("/"),
        )

    # Important assertions
    assert cfg.TRAIN.DATASET.lower() == args.dataset.lower()
    assert (
        cfg.ACTIVE_LEARNING.BUDGET_SIZE == args.budget_size
    ), f"args.budget_size: {args.budget_size} and cfg budgetsize: {cfg.ACTIVE_LEARNING.BUDGET_SIZE} should be same."
    assert cfg.MODEL.NUM_CLASSES == args.num_classes

    if args.reuse_aml:
        prefix_dir, suffix_dir = cfg.OUT_DIR.split("start_")
        cfg.OUT_DIR = os.path.join(prefix_dir, f"start_{args.seed_id}", suffix_dir[2:])

    # Skip if index sets are already saved
    if not check_path_exists(
        check_dir=cfg.OUT_DIR, fnames=["lSet", "uSet", "activeSet"]
    ):
        # Saves the index sets
        active_sampling(cfg, debug=True)
    else:
        print("--------------------")
        print(
            f"Skipping best model inference as index sets already exists at {cfg.OUT_DIR}"
        )

    print(f"cfg.OUT_DIR: {cfg.OUT_DIR}")
    print(f"cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR: {cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR}")
    return cfg.OUT_DIR, cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR


def check_path_exists(
    check_dir: str, fnames: list, file_extension: str = ".npy"
) -> bool:
    """Checks whether given filenames exist at given directory.

    Args:
        check_dir (str): [description]
        fnames (list): [description]
        file_extension (str, optional): [description]. Defaults to '.npy'.

    Returns:
        bool: Returns whether file exist or not.
    """

    assert type(check_dir) == str
    assert type(fnames) == list
    assert type(file_extension) == str

    assert len(fnames) > 0

    for name in fnames:
        temp_path = os.path.join(check_dir, name + file_extension)
        if not os.path.isfile(temp_path):
            return False

    return True
