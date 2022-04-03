#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions that handle saving and loading of checkpoints."""

import os
import torch

# from pycls.core.config import cfg

import pycls.utils.distributed as du

# from pycls.core.config import dump_cfg
import copy

# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"
# Checkpoints directory name
_DIR_NAME = "checkpoints"


def get_checkpoint_dir_wo_checkpoint(cfg):
    temp_dir_name = _DIR_NAME
    print("cfg.OUT_DIR: ", cfg.OUT_DIR)
    return os.path.join(cfg.OUT_DIR, temp_dir_name)


def get_checkpoint_dir(cfg):
    """Retrieves the location for storing checkpoints."""
    temp_dir_name = _DIR_NAME
    return os.path.join(cfg.OUT_DIR, temp_dir_name)


def get_checkpoint(cfg, epoch):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(cfg), name)


def get_last_checkpoint(cfg):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir(cfg)
    # Checkpoint file names are in lexicographic order
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint(cfg):
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir(cfg)
    if not os.path.exists(checkpoint_dir):
        return False
    return any([_NAME_PREFIX in f for f in os.listdir(checkpoint_dir)])


def is_checkpoint_epoch(cfg, cur_epoch):
    """Determines if a checkpoint should be saved on current epoch."""
    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_checkpoint(cfg, model, optimizer, epoch):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if not du.is_master_proc(cfg):
        return
    # Ensure that the checkpoint dir exists
    os.makedirs(get_checkpoint_dir(cfg), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = get_checkpoint(cfg, epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def state_save_checkpoint(cfg, info, model_state, optimizer_state, epoch):
    """Saves a checkpoint with different states"""
    if not du.is_master_proc(cfg):
        return
    # Ensure that the checkpoint dir exists
    os.makedirs(get_checkpoint_dir(cfg), exist_ok=True)

    sd = model_state
    opt_state = optimizer_state
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": opt_state,
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    global _NAME_PREFIX
    _NAME_PREFIX = info + "_" + _NAME_PREFIX

    checkpoint_file = get_checkpoint(cfg, epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    print("Model checkpoint saved at path: {}".format(checkpoint_file))
    _NAME_PREFIX = "model_epoch_"
    return checkpoint_file


def named_save_checkpoint(cfg, info, model, optimizer, epoch):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if not du.is_master_proc(cfg):
        return
    # Ensure that the checkpoint dir exists
    os.makedirs(get_checkpoint_dir(cfg), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    global _NAME_PREFIX
    _NAME_PREFIX = info + "_" + _NAME_PREFIX
    checkpoint_file = get_checkpoint(cfg, epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    print("Model checkpoint saved at path: {}".format(checkpoint_file))

    _NAME_PREFIX = "model_epoch_"
    return checkpoint_file


def load_checkpoint(cfg, checkpoint_file, model, optimizer=None, active_sampling=False):
    """Loads the checkpoint from the given file."""
    assert os.path.exists(checkpoint_file), "Checkpoint '{}' not found".format(
        checkpoint_file
    )

    # Load the checkpoint on CPU to avoid GPU mem spike
    temp_checkpoint = torch.load(checkpoint_file, map_location="cpu")
    checkpoint = copy.deepcopy(temp_checkpoint)
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model
    # if not active_sampling:

    print("==============================")
    print("cfg.NUM_GPUS: ", cfg.NUM_GPUS)
    print("==============================")
    ms = model.module if cfg.NUM_GPUS > 1 else model

    isModuleStrPresent = False
    if "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]

    for k in checkpoint.keys():
        if k.find("module.") == -1:
            continue
        isModuleStrPresent = True
        break

    # remove module
    if isModuleStrPresent:
        print("Loaded checkpoint contains module present in keys.")
        print("So now removing 'module' strings")
        # remove module strings
        from collections import OrderedDict

        new_ckpt_dict = OrderedDict()
        for k, v in checkpoint.items():
            tmp_key = k.replace("module.", "")
            new_ckpt_dict[tmp_key] = v

        checkpoint = copy.deepcopy(new_ckpt_dict)
        print("Done!!")

    ms.load_state_dict(checkpoint)
    ms.cuda(torch.cuda.current_device())

    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(temp_checkpoint["optimizer_state"])
    return 0 if isModuleStrPresent else temp_checkpoint["epoch"]
