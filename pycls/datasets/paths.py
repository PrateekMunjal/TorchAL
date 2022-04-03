#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset paths."""

import os
from pycls.core.config import cfg

# Default data directory (/path/pycls/pycls/datasets/data)
_DEF_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Data paths
_paths = {"cifar10": cfg.TRAIN_DIR, "IMAGENET": cfg.TRAIN_DIR}


def has_data_path(dataset_name):
    """Determines if the dataset has a data path."""
    return dataset_name in _paths.keys()


def get_data_path(dataset_name):
    """Retrieves data path for the dataset."""
    print("------------Inside get_data_path in paths.py-----------")
    print("Path: {}".format(_paths[dataset_name]))
    print("--------------------------------------------------------")
    return _paths[dataset_name]


def register_path(name, path):
    """Registers a dataset path dynamically."""
    _paths[name] = path
