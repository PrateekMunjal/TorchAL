#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import os

from yacs.config import CfgNode as CN


# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

## SIMPLE AUGMENTATIONS
_C.SIMPLE_AUGMENTATIONS = True
_C.TRAIN_DIR = ""
_C.TEST_DIR = ""
_C.DIR_SPECIFIC = "vanilla"

# ------------------------------------------------------------------------------#
# VAAL Options
# ------------------------------------------------------------------------------#
_C.VAAL = CN()

_C.VAAL.TRAIN_VAAL = False
_C.VAAL.Z_DIM = 32
_C.VAAL.VAE_BS = 64
_C.VAAL.VAE_EPOCHS = 15
_C.VAAL.VAE_LR = 5e-4
_C.VAAL.DISC_LR = 5e-4
_C.VAAL.BETA = 1.0
_C.VAAL.ADVERSARY_PARAM = 1.0
_C.VAAL.IM_SIZE = 32


# ---------------------------------------------------------------------
# Shake Shake Regularization Options
# ---------------------------------------------------------------------
_C.SHAKE_SHAKE = CN()

_C.SHAKE_SHAKE.BASE_CHANNELS = 32

_C.SHAKE_SHAKE.DEPTH = 26

_C.SHAKE_SHAKE.FORWARD = True

_C.SHAKE_SHAKE.BACKWARD = True

_C.SHAKE_SHAKE.IMAGE = True

# ------------------------------------------------------------------------------#
# Ensemble Options
# ------------------------------------------------------------------------------#
_C.ENSEMBLE = CN()

_C.ENSEMBLE.NUM_MODELS = 3

_C.ENSEMBLE.SAME_MODEL = True

_C.ENSEMBLE.MODEL_TYPE = ["resnet_1"]

_C.ENSEMBLE.MAX_EPOCH = 1

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Model type
_C.MODEL.TYPE = "vgg"

# Number of weight layers
_C.MODEL.DEPTH = 16

# Number of classes
_C.MODEL.NUM_CLASSES = 10

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSS_FUN = "cross_entropy"

_C.MODEL.TRANSFER_MODEL_STYLE = ""

_C.MODEL.TRANSFER_MODEL_TYPE = ""

# _C.MODEL.TRANSFER_DATA_SPLIT = 10

# Number of weight layers
_C.MODEL.TRANSFER_MODEL_DEPTH = 0

_C.MODEL.TRANSFER_DIR_SPECIFIC = "vanilla"

# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
_C.RESNET = CN()

# Transformation function (see pycls/models/resnet.py for options)
_C.RESNET.TRANS_FUN = "basic_transform"

# Number of groups to use (1 -> ResNet; > 1 -> ResNeXt)
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt)
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply stride to 1x1 conv (True -> MSRA; False -> fb.torch)
_C.RESNET.STRIDE_1X1 = False


# ---------------------------------------------------------------------------- #
# VGG options
# ---------------------------------------------------------------------------- #
_C.VGG = CN()

# Transformation function (see pycls/models/resnet.py for options)
_C.VGG.TRANS_FUN = "basic_transform"

# Number of groups to use (1 -> ResNet; > 1 -> ResNeXt)
_C.VGG.NUM_GROUPS = 1

# ---------------------------------------------------------------------------- #
# AnyNet options
# ---------------------------------------------------------------------------- #
_C.ANYNET = CN()

# Stem type
_C.ANYNET.STEM_TYPE = "plain_block"

# Stem width
_C.ANYNET.STEM_W = 32

# Block type
_C.ANYNET.BLOCK_TYPE = "plain_block"

# Depth for each stage (number of blocks in the stage)
_C.ANYNET.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.ANYNET.WIDTHS = []

# Strides for each stage (applies to the first block of each stage)
_C.ANYNET.STRIDES = []

# Bottleneck multipliers for each stage (applies to bottleneck block)
_C.ANYNET.BOT_MULS = []

# Number of groups for each stage (applies to bottleneck block)
_C.ANYNET.NUM_GS = []


# ---------------------------------------------------------------------------- #
# EfficientNet options
# ---------------------------------------------------------------------------- #
_C.EN = CN()

# Stem width
_C.EN.STEM_W = 32

# Depth for each stage (number of blocks in the stage)
_C.EN.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.EN.WIDTHS = []

# Expansion ratios for MBConv blocks in each stage
_C.EN.EXP_RATIOS = []

# Squeeze-and-Excitation (SE) operation
_C.EN.SE_ENABLED = True

# Squeeze-and-Excitation (SE) ratio
_C.EN.SE_RATIO = 0.25

# Strides for each stage (applies to the first block of each stage)
_C.EN.STRIDES = []

# Kernel sizes for each stage
_C.EN.KERNELS = []

# Head width
_C.EN.HEAD_W = 1280

# Drop connect ratio
_C.EN.DC_RATIO = 0.0

# Dropout ratio
_C.EN.DROPOUT_RATIO = 0.0


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()

_C.OPTIM.TYPE = "sgd"

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

# Exponential decay factor
_C.OPTIM.GAMMA = 0.1

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = False

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# ---------------------------------------------------------------------------- #
# TOD options
# ---------------------------------------------------------------------------- #
_C.TOD = CN()

_C.TOD.CKPT_PATH = ""

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

_C.TRAIN.IMBALANCED = False

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Image size
_C.TRAIN.IM_SIZE = 224

_C.TRAIN.IM_CHANNELS = 3

# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = False

# Weights to start training from
_C.TRAIN.WEIGHTS = ""

_C.TRAIN.TRANSFER_EXP = False

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Dataset and split
_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 50

# Image size
_C.TEST.IM_SIZE = 256

# Weights to use for testing
_C.TEST.WEIGHTS = ""


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers per training process
_C.DATA_LOADER.NUM_WORKERS = 4

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# Memory options
# ---------------------------------------------------------------------------- #
_C.MEM = CN()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True


# ---------------------------------------------------------------------------- #
# CUDNN options
# ---------------------------------------------------------------------------- #
_C.CUDNN = CN()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = False


# ---------------------------------------------------------------------------- #
# Precise timing options
# ---------------------------------------------------------------------------- #
_C.PREC_TIME = CN()

# Perform precise timing at the start of training
_C.PREC_TIME.ENABLED = False

# Total mini-batch size
_C.PREC_TIME.BATCH_SIZE = 128

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 30


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Output directory
_C.OUT_DIR = "/tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 40

# Distributed backend
_C.DIST_BACKEND = "nccl"

# Hostname and port for initializing multi-process groups
_C.HOST = "localhost"
_C.PORT = 10001

# ---------------------------------------------------------------------------- #
# RANDAUG options
# ---------------------------------------------------------------------------- #
_C.RANDAUG = CN()

_C.RANDAUG.ACTIVATE = False

_C.RANDAUG.N = 1

_C.RANDAUG.M = 5

# #-------------------------------------------------------------------------------#
# #  ACTIVE LEARNING options
# #-------------------------------------------------------------------------------#
_C.ACTIVE_LEARNING = CN()

_C.ACTIVE_LEARNING.SAMPLING_FN = "random"

_C.ACTIVE_LEARNING.ACTIVATE = False

_C.ACTIVE_LEARNING.AL_ITERATION = 0

_C.ACTIVE_LEARNING.LSET_PATH = ""

_C.ACTIVE_LEARNING.USET_PATH = ""

_C.ACTIVE_LEARNING.VALSET_PATH = ""

_C.ACTIVE_LEARNING.MODEL_LOAD_DIR = ""

_C.ACTIVE_LEARNING.MODEL_SAVE_DIR = ""

_C.ACTIVE_LEARNING.DATA_SPLIT = 0.0

_C.ACTIVE_LEARNING.BUDGET_SIZE = 64058  # 10% of initial lSet

_C.ACTIVE_LEARNING.N_BINS = 500  # Used by UC_uniform

_C.ACTIVE_LEARNING.DROPOUT_ITERATIONS = 0  # Used by DBAL

_C.ACTIVE_LEARNING.NOISY_ORACLE = 0.0
# _C.ACTIVE_LEARNING.RESULTS_DIR = './results_al/'

# _C.ACTIVE_LEARNING.PARENT_DIR = '.'

# _C.ACTIVE_LEARNING.MODEL_EPOCH = 0

# _C.ACTIVE_LEARNING.VAL_ACCURACY = 0.

# #-------------------------------------------------------------------------------#
# #  SWA mode options
# #-------------------------------------------------------------------------------#
_C.SWA_MODE = CN()

_C.SWA_MODE.ACTIVATE = False

_C.SWA_MODE.START_ITER = 0

_C.SWA_MODE.FREQ = 50

_C.SWA_MODE.LR = 5e-4


def assert_cfg():
    """Checks config values invariants."""
    assert (
        not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0
    ), "The first lr step must start at 0"
    assert _C.TRAIN.SPLIT in [
        "train",
        "val",
        "test",
    ], "Train split '{}' not supported".format(_C.TRAIN.SPLIT)
    assert (
        _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0
    ), "Train mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TEST.SPLIT in [
        "train",
        "val",
        "test",
    ], "Test split '{}' not supported".format(_C.TEST.SPLIT)
    assert (
        _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0
    ), "Test mini-batch size should be a multiple of NUM_GPUS."
    assert (
        not _C.BN.USE_PRECISE_STATS or _C.NUM_GPUS == 1
    ), "Precise BN stats computation not verified for > 1 GPU"
    assert _C.LOG_DEST in [
        "stdout",
        "file",
    ], "Log destination '{}' not supported".format(_C.LOG_DEST)
    assert (
        not _C.PREC_TIME.ENABLED or _C.NUM_GPUS == 1
    ), "Precise iter time computation not verified for > 1 GPU"

    # our assertions
    if _C.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty_uniform_discretize":
        assert (
            _C.ACTIVE_LEARNING.N_BINS != 0
        ), "The number of bins used cannot be 0. Please provide a number >0 for {} sampling function".format(
            _C.ACTIVE_LEARNING.SAMPLING_FN
        )


def custom_assert_cfg(temp_cfg):
    """Checks config values invariants."""
    assert (
        not temp_cfg.OPTIM.STEPS or temp_cfg.OPTIM.STEPS[0] == 0
    ), "The first lr step must start at 0"
    assert temp_cfg.TRAIN.SPLIT in [
        "train",
        "val",
        "test",
    ], "Train split '{}' not supported".format(temp_cfg.TRAIN.SPLIT)
    assert (
        temp_cfg.TRAIN.BATCH_SIZE % temp_cfg.NUM_GPUS == 0
    ), "Train mini-batch size should be a multiple of NUM_GPUS."
    assert temp_cfg.TEST.SPLIT in [
        "train",
        "val",
        "test",
    ], "Test split '{}' not supported".format(temp_cfg.TEST.SPLIT)
    assert (
        temp_cfg.TEST.BATCH_SIZE % temp_cfg.NUM_GPUS == 0
    ), "Test mini-batch size should be a multiple of NUM_GPUS."
    assert (
        not temp_cfg.BN.USE_PRECISE_STATS or temp_cfg.NUM_GPUS == 1
    ), "Precise BN stats computation not verified for > 1 GPU"
    assert temp_cfg.LOG_DEST in [
        "stdout",
        "file",
    ], "Log destination '{}' not supported".format(temp_cfg.LOG_DEST)
    assert (
        not temp_cfg.PREC_TIME.ENABLED or temp_cfg.NUM_GPUS == 1
    ), "Precise iter time computation not verified for > 1 GPU"

    # our assertions
    if temp_cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty_uniform_discretize":
        assert (
            temp_cfg.ACTIVE_LEARNING.N_BINS != 0
        ), "The number of bins used cannot be 0. Please provide a number >0 for {} sampling function".format(
            temp_cfg.ACTIVE_LEARNING.SAMPLING_FN
        )


def custom_dump_cfg(temp_cfg):
    """Dumps the config to the output directory."""
    print("===========In custom dum cfg===========")
    print(
        f"temp_cfg.OUT_DIR: {temp_cfg.OUT_DIR}, temp_cfg.CFG_DEST: {temp_cfg.CFG_DEST}"
    )
    print("=======================================")
    os.makedirs(temp_cfg.OUT_DIR, exist_ok=True)
    cfg_file = os.path.join(temp_cfg.OUT_DIR, temp_cfg.CFG_DEST)
    with open(cfg_file, "w") as f:
        # _C.dump(stream=f)
        temp_cfg.dump(stream=f)


def load_custom_cfg(temp_cfg, out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory into temp_cfg."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    temp_cfg.merge_from_file(cfg_file)
    return temp_cfg


def dump_cfg(cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(cfg.OUT_DIR, cfg.CFG_DEST)
    with open(cfg_file, "w") as f:
        cfg.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)
