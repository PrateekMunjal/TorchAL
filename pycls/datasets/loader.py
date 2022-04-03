#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

from torch.utils.data.distributed import DistributedSampler

from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from pycls.datasets.sampler import IndexedSequentialSampler, IndexedDistributedSampler

import torch

from pycls.datasets.cifar10 import Cifar10
from pycls.datasets.imagenet import ImageNet

import pycls.datasets.paths as dp
import numpy as np
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

# Supported datasets
_DATASET_CATALOG = {"cifar10": Cifar10, "IMAGENET": ImageNet}


def loadPartitions(lSetPath, uSetPath, valSetPath):
    assert isinstance(lSetPath, str), "Expected lSetPath to be a string."
    assert isinstance(uSetPath, str), "Expected uSetPath to be a string."
    assert isinstance(valSetPath, str), "Expected lSetPath to be a string."

    lSet = np.load(lSetPath, allow_pickle=True)
    uSet = np.load(uSetPath, allow_pickle=True)
    valSet = np.load(valSetPath, allow_pickle=True)

    # Checking no overlap
    assert (
        len(set(valSet) & set(uSet)) == 0
    ), "Intersection is not allowed between validationset and uset"
    assert (
        len(set(valSet) & set(lSet)) == 0
    ), "Intersection is not allowed between validationset and lSet"
    assert (
        len(set(uSet) & set(lSet)) == 0
    ), "Intersection is not allowed between uSet and lSet"

    print("========== index sets loaded successfully ============")
    print(f"Lset: {len(lSet)} | uSet: {len(uSet)} | valSet: {len(valSet)}")
    print("=====================================================")

    return lSet, uSet, valSet


def _construct_loader(
    cfg,
    dataset_name,
    split,
    batch_size,
    shuffle,
    drop_last,
    indexSet=None,
    isAug=True,
    isDistributed=True,
    isVaalSampling=False,
    allowRepeat=True,
):
    """Constructs the data loader for the given dataset."""
    assert dataset_name in _DATASET_CATALOG.keys(), "Dataset '{}' not supported".format(
        dataset_name
    )
    assert dp.has_data_path(dataset_name), "Dataset '{}' has no data path".format(
        dataset_name
    )
    # Retrieve the data path for the dataset
    data_path = cfg.TRAIN_DIR
    # dp.get_data_path(dataset_name)

    # Construct the dataset
    dataset = _DATASET_CATALOG[dataset_name](
        cfg, data_path, split, isAug, isVaalSampling
    )

    # Create a sampler for multi-process training
    if isDistributed:
        if indexSet is None:
            sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
        else:
            sampler = (
                IndexedDistributedSampler(
                    dataset=dataset, index_set=indexSet, allowRepeat=allowRepeat
                )
                if cfg.NUM_GPUS > 1
                else None
            )
            print("=============================")
            print("IndexedDistributedSampler data sampler")
            print("sampler: {}".format(sampler))
            print("=============================")
    else:
        # Called when we are in some subprocess or when we simply want to use torch dataloader
        assert (
            len(indexSet) != 0
        ), f"IndexSet cannot be empty. Currently len(indexSet): {len(indexSet)}"
        if shuffle:
            sampler = SubsetRandomSampler(indexSet)
        else:
            sampler = IndexedSequentialSampler(indexSet)

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_loader_no_aug(
    cfg, indices=None, isDistributed=True, isShuffle=True, isVaalSampling=False
):
    print("====== Constructing dataloader no Augmentation ======")
    print("isVaalSampling: {}".format(isVaalSampling))
    temp_bs = (
        int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        if isDistributed
        else cfg.TRAIN.BATCH_SIZE
    )
    if isVaalSampling:
        temp_bs = cfg.VAAL.VAE_BS
    print(
        f"Dataset_name: [{cfg.TRAIN.DATASET}], split: [{cfg.TRAIN.SPLIT}], batch_size: [{temp_bs}]"
    )

    return _construct_loader(
        cfg,
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=temp_bs,
        shuffle=isShuffle,
        drop_last=False,
        indexSet=indices,
        isAug=False,  # Set to True if you want augmentations to be on while iterating
        isDistributed=isDistributed,
        isVaalSampling=isVaalSampling,
    )


def construct_train_loader(cfg, indices=None, isDistributed=True):
    """Train loader wrapper."""
    print("====== Constructing dataloader ======")
    print(
        f"Dataset_name: [{cfg.TRAIN.DATASET}], split: [{cfg.TRAIN.SPLIT}], batch_size: [{cfg.TRAIN.BATCH_SIZE//cfg.NUM_GPUS}]"
    )
    return _construct_loader(
        cfg,
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        if isDistributed
        else cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        indexSet=indices,
        isDistributed=isDistributed,
    )


def construct_val_loader(cfg, indices=None, isShuffle=False, isDistributed=True):
    """Val loader wrapper."""
    return _construct_loader(
        cfg,
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
        if isDistributed
        else cfg.TEST.BATCH_SIZE,
        shuffle=isShuffle,
        drop_last=False,
        indexSet=indices,
        isAug=False,  # Set to True if you want augmentations to be on while iterating through validation set
        isDistributed=isDistributed,
        allowRepeat=False,
    )


def construct_test_loader(cfg, indices=None):
    """Test loader wrapper."""
    return _construct_loader(
        cfg,
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
        indexSet=indices,
        isAug=False,
        allowRepeat=False,
    )


def get_random_subset(x, p):
    """
    x: np.array
    p: proportion
    """
    x_len = len(x)
    indexes = np.arange(x_len)
    np.random.seed(0)
    np.random.shuffle(indexes)
    stopIndex = int(x_len * p)
    return x[indexes[:stopIndex]]


def get_luset_data_loaders(cfg):
    """
    Get lSet and uSet data loaders.
    """
    lSetPath = cfg.ACTIVE_LEARNING.LSET_PATH
    uSetPath = cfg.ACTIVE_LEARNING.USET_PATH
    valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH

    lSet, uSet, _ = loadPartitions(
        lSetPath=lSetPath, uSetPath=uSetPath, valSetPath=valSetPath
    )

    lSetLoader = construct_loader_no_aug(cfg, indices=lSet)
    uSetLoader = construct_loader_no_aug(cfg, indices=uSet)

    return lSetLoader, uSetLoader


def get_data_loaders(cfg, isDistributed=True):
    """get train, val and test data loaders"""

    lSetPath = cfg.ACTIVE_LEARNING.LSET_PATH
    uSetPath = cfg.ACTIVE_LEARNING.USET_PATH
    valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH

    lSet, uSet, valSet = loadPartitions(
        lSetPath=lSetPath, uSetPath=uSetPath, valSetPath=valSetPath
    )

    lSetLoader = construct_train_loader(cfg, indices=lSet, isDistributed=isDistributed)

    # Using 10% validation set
    reduced_valSet = get_random_subset(x=valSet, p=0.1)
    print("-----------PARTITIONS LOADED--------------")
    print(
        "PATHS: LSET[{}], USET[{}], VALSET[{}]".format(lSetPath, uSetPath, valSetPath)
    )
    print("lSet: {}, uSet: {}, valSet:{}".format(len(lSet), len(uSet), len(valSet)))
    print("reduced valSet size used for validation: {}".format(len(reduced_valSet)))
    print("------------------------------------------")

    valSetLoader = construct_val_loader(
        cfg, indices=reduced_valSet, isDistributed=isDistributed
    )

    return lSetLoader, valSetLoader


def shuffle(loader, cur_epoch):
    """ "Shuffles the data."""
    assert isinstance(
        loader.sampler,
        (
            SubsetRandomSampler,
            IndexedSequentialSampler,
            RandomSampler,
            DistributedSampler,
            IndexedDistributedSampler,
        ),
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler) or isinstance(
        loader.sampler, IndexedDistributedSampler
    ):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
