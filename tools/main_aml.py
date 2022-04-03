#!/usr/bin/env python3

# Copyright (c) Prateek Munjal.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from pycls.core.config import cfg
from pycls.core.config import custom_dump_cfg
from automl_args import args

from helper.aml_utils import automl_model_inference
from helper.subprocess_utils import (
    auto_ml_sp,
    swa_on_auto_ml_sp,
    test_net_subprocess_call,
)

from copy import deepcopy
import os

# 5 steps
# 1. Train auto_ml models; followed by SWA;                                                                                                      [DONE]
# 2. Search best model from auto_ml models saved during random trials                                                                            [DONE]
# 3. Test the best model from step 2                                                                                                             [DONE]
# 4. Inference using the best model                                                                                                              [DONE]
# 5. Train auto_ml models; followed by SWA; -- for sampling functions -- results will be saved as "automl/model_type_depth/CIFAR10/20.0/random/" [DONE]
# Loop from 2 to 5 for active learning iterations


def get_directory_specific(args) -> str:
    """Return directory named based on which strong-regularization technique is applied.

    Args:
        args: Reference to input args.

    Returns:
        str: Strong-regularization identifier.
    """
    directory_specific = "vanilla"

    if args.swa_mode and args.rand_aug:
        directory_specific = "swa_rand_aug"
    elif args.swa_mode:
        directory_specific = "swa"
    elif args.rand_aug:
        directory_specific = "rand_aug"
    else:
        print("========= [NO ADVANCED REGULARIZATION TRICK ACTIVATED] =========")

    if args.isTransferExp:
        directory_specific = args.transfer_dir_specific

    return directory_specific


def main(args):
    """Main function implementing AL setting"""

    curr_partition = args.init_partition
    curr_model_type = args.model_type
    curr_model_depth = args.model_depth

    os.makedirs(args.out_dir, exist_ok=True)

    curr_directory_specific = get_directory_specific(args)

    if args.reuse_aml:
        automl_path = os.path.join(
            args.out_dir,
            "auto_ml_results",
            f"lSet_{str(args.lSet_partition)}",
            f"start_{str(args.reuse_aml_seed)}",
            args.dataset,
            str(curr_partition),
            f"{curr_model_type}_depth_{curr_model_depth}",
            curr_directory_specific,
        )
    else:
        automl_path = os.path.join(
            args.out_dir,
            "auto_ml_results",
            f"lSet_{str(args.lSet_partition)}",
            f"start_{str(args.seed_id)}",
            args.dataset,
            str(curr_partition),
            f"{curr_model_type}_depth_{curr_model_depth}",
            curr_directory_specific,
        )

    for curr_al_iteration in range(1, args.al_max_iter + 1):

        cfg.ACTIVE_LEARNING.AL_ITERATION = curr_al_iteration

        # run_automl_curr_iter = False
        if args.isTransferExp:
            cfg.TRAIN.TRANSFER_EXP = args.isTransferExp
            cfg.MODEL.TRANSFER_MODEL_TYPE = args.transfer_model_type
            cfg.MODEL.TRANSFER_MODEL_DEPTH = args.transfer_model_depth
            cfg.MODEL.TRANSFER_MODEL_STYLE = args.transfer_model_style
            cfg.MODEL.TRANSFER_DIR_SPECIFIC = args.transfer_dir_specific

        if curr_al_iteration == 1:
            # the very first iteration so initialize cfg file
            assert os.path.isfile(args.cfg_file)
            cfg.merge_from_file(args.cfg_file)
            cfg.DIR_SPECIFIC = curr_directory_specific
            cfg.OUT_DIR = automl_path
            cfg.OPTIM.MAX_EPOCH = args.clf_epochs
            cfg.ACTIVE_LEARNING.ACTIVATE = False
            cfg.TRAIN_DIR = args.train_dir
            cfg.TEST_DIR = args.test_dir
            cfg.ACTIVE_LEARNING.LSET_PATH = args.lSetPath
            cfg.ACTIVE_LEARNING.USET_PATH = args.uSetPath
            cfg.ACTIVE_LEARNING.VALSET_PATH = args.valSetPath
            cfg.ACTIVE_LEARNING.DATA_SPLIT = args.init_partition
            cfg.PORT = args.port
            cfg.NUM_GPUS = args.n_GPU
            cfg.DIST_BACKEND = "nccl"
            cfg.RNG_SEED = args.seed_id
            cfg.MODEL.NUM_CLASSES = args.num_classes
            # imbalanced exps
            cfg.TRAIN.IMBALANCED = args.isimbalanced
            # RA
            cfg.RANDAUG.ACTIVATE = args.rand_aug

            cfg.TRAIN.EVAL_PERIOD = args.eval_period

            # NOISY EXPS
            cfg.ACTIVE_LEARNING.NOISY_ORACLE = args.noisy_oracle

            # SWA args
            cfg.SWA_MODE.ACTIVATE = args.swa_mode
            cfg.SWA_MODE.FREQ = args.swa_freq
            cfg.SWA_MODE.LR = args.swa_lr
            cfg.SWA_MODE.START_ITER = args.swa_iter

        print("~~~ out_dir: ", cfg.OUT_DIR)
        check_prev_subpath, check_next_subpath = (cfg.OUT_DIR).split("auto_ml_results")
        if check_next_subpath[0] == "/":
            check_next_subpath = check_next_subpath[1:]
        check_aml_path = os.path.join(
            check_prev_subpath, "best_automl_results", check_next_subpath, "checkpoints"
        )
        print("check_aml_path: ")
        print(check_aml_path)

        if not os.path.exists(check_aml_path):

            auto_ml_sp(cfg, args, debug=True)

        else:

            print("Auto ml already exists; So skip doing automl for this!")

        # Apply SWA on auto-ml models
        if cfg.DIR_SPECIFIC == "swa_rand_aug" or cfg.DIR_SPECIFIC == "swa":
            print("Applying SWA...!!")
            swa_on_auto_ml_sp(cfg, args, debug=True)

        automl_path = cfg.OUT_DIR

        cfg_dir, best_model_path = automl_model_inference(
            args, cfg, automl_path, al_iteration=curr_al_iteration
        )

        best_model_cfg_dir = best_model_path.split("checkpoints")[0]

        print(f"Passing best model_cfg_dir: {best_model_cfg_dir}")
        print(f"cfg_dir: {cfg_dir}")

        os.makedirs(cfg_dir, exist_ok=True)

        best_model_test_acc = test_net_subprocess_call(
            best_model_cfg_dir, best_model_path, debug=True
        )

        # Get new config file
        #
        if args.isTransferExp and not os.path.isfile(
            os.path.join(cfg_dir, "config.yaml")
        ):

            temp_cfg = deepcopy(cfg)
            temp_cfg.ACTIVE_LEARNING.ACTIVATE = True
            temp_cfg.ACTIVE_LEARNING.LSET_PATH = os.path.join(cfg.OUT_DIR, "lSet.npy")
            temp_cfg.ACTIVE_LEARNING.USET_PATH = os.path.join(cfg.OUT_DIR, "uSet.npy")
            custom_dump_cfg(temp_cfg)

        cfg.merge_from_file(os.path.join(cfg_dir, "config.yaml"))

        # set output dir because now we need to save auto-ml models
        prev_out_dir = cfg.OUT_DIR
        print("prev_out_dir i.e cfg.OUT_DIR[old]: ", prev_out_dir)

        tochange_part_dir, unchanged_part_dir = cfg.OUT_DIR.split("lSet_")
        unchanged_part_dir = "lSet_" + unchanged_part_dir
        tochange_part_dir = os.path.join(args.out_dir, "auto_ml_results")
        print(f"cfg.OUT_DIR[old]: {cfg.OUT_DIR}")
        cfg.OUT_DIR = os.path.join(tochange_part_dir, unchanged_part_dir)
        print(f"cfg.OUT_DIR[new]: {cfg.OUT_DIR}")


if __name__ == "__main__":
    main(args)
