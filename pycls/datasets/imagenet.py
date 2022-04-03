#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import cv2
import numpy as np
import os
import re
import torch
import torch.utils.data


import pycls.datasets.transforms as transforms
import pycls.utils.logging as lu
from .autoaugment import RandAugmentPolicy

logger = lu.get_logger(__name__)

## NEW _MEAN and _SD in RGB mode as PIL image mode tells image in RGB mode.
_MEAN = [0.0484, 0.0454, 0.0403]
_SD = [0.0225, 0.022, 0.022]


# Eig vals and vecs of the cov mat
_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)


class ImageNet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, cfg, data_path, split, isAug=True, isVaalSampling=False):
        print("-------------------------")
        logger.info("data_path: {}".format(data_path))
        print("-------------------------")
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        assert split in [
            "train",
            "val",
        ], "Split '{}' not supported for ImageNet".format(split)
        logger.info("Constructing ImageNet {}...".format(split))
        self._data_path = data_path
        self._split = split
        self.cfg = cfg
        self._is_aug = isAug
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self._data_path, self._split)
        logger.info("{} data path: {}".format(self._split, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        self._class_ids = sorted(
            [f for f in os.listdir(split_path) if re.match(r"^n[0-9]+$", f)]
        )
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in os.listdir(im_dir):
                self._imdb.append(
                    {
                        "im_path": os.path.join(im_dir, im_name),
                        "class": cont_id,
                    }
                )
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        # Train and test setups differ
        if self._split == "train" and self._is_aug:
            # Scale and aspect ratio
            # temp_im_target_size = self.cfg.VAAL.IM_SIZE if isVaalSampling else self.cfg.TRAIN.IM_SIZE
            temp_im_target_size = self.cfg.TRAIN.IM_SIZE

            im = transforms.random_sized_crop(
                im=im, size=temp_im_target_size, area_frac=0.08
            )
            # Horizontal flip
            im = transforms.horizontal_flip(im=im, p=0.5)

            if self.cfg.RANDAUG.ACTIVATE:
                # Apply randAugmentation here
                # converting to PIL image
                im = transforms.getPILImage(im.astype(np.uint8))
                # print("im.mode: {}".format(im.mode))

                # RandAug
                randAug = RandAugmentPolicy(N=self.cfg.RANDAUG.N, M=self.cfg.RANDAUG.M)
                im = randAug(im)
                # converting back to np array
                im = np.array(im)
                # print("After applying randAug")

        else:
            # Scale and center crop
            # print("in test mode with split: {}".format(self._split))
            im = transforms.scale(self.cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(self.cfg.TRAIN.IM_SIZE, im)
        # HWC -> CHW
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # I am unsure for this cov matrices eig vals and eig vecs
        # # PCA jitter
        # if self._split == 'train':
        #     im = transforms.lighting(im, 0.1, _EIG_VALS, _EIG_VECS)

        # Color normalization
        if self.cfg.TRAIN.IM_SIZE == self.cfg.VAAL.IM_SIZE:
            # Dont standardize for VAE and disc learning
            pass
        else:
            im = transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        # Load the image
        im = cv2.imread(self._imdb[index]["im_path"])
        # convert img from BGR to RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32, copy=False)
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        # Retrieve the label
        label = self._imdb[index]["class"]
        return im.astype(np.float32, copy=False), label

    def __len__(self):
        return len(self._imdb)
