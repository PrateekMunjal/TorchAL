from al_utils.vaal_util import train_vae, train_vae_disc
from al_utils import vae_sampling as vs
import sys
import pickle
import torch
import numpy as np
import os
from copy import deepcopy
from pycls.core.config import custom_dump_cfg

import pycls.datasets.loader as imagenet_loader


def save_numpy_arrays(arrays, names, parent_dir, saveinText=False):
    """Saves numpy arrays"""
    for i, a in enumerate(arrays):
        if saveinText:
            np.savetxt(parent_dir + names[i] + ".txt", a, fmt="%d")
            print(
                "Saved {} at path: {} !!".format(
                    names[i], parent_dir + names[i] + ".txt"
                )
            )
        else:
            np.save(parent_dir + names[i] + ".npy", a)
            print(
                "Saved {} at path: {} !!".format(
                    names[i], parent_dir + names[i] + ".npy"
                )
            )


#     #train task model


def vaal_active_sampling(cfg, dataObj, debug=False):
    """Implements VAAL sampling.

    Args:
        cfg: Reference to the config yaml
        dataObj: Reference to data class
        debug (bool, optional): Switch for debug mode. Defaults to False.
    """

    temp_old_im_size = cfg.TRAIN.IM_SIZE
    if cfg.TRAIN.DATASET.lower() == "imagenet":
        cfg.TRAIN.IM_SIZE = cfg.VAAL.IM_SIZE  # args.vaal_im_size
        print("cfg.TRAIN.IM_SIZE: ", cfg.TRAIN.IM_SIZE)
        print("cfg.VAAL.IM_SIZE: ", cfg.VAAL.IM_SIZE)

    lSet_path = cfg.ACTIVE_LEARNING.LSET_PATH
    uSet_path = cfg.ACTIVE_LEARNING.USET_PATH

    if debug:
        print("lSetPath: {}".format(lSet_path))
    if debug:
        print("uSetPath: {}".format(uSet_path))

    lSet = np.load(lSet_path, allow_pickle=True)
    uSet = np.load(uSet_path, allow_pickle=True)

    print("---------Loaded partitions--------")
    print("lSet: {}, uSet: {}".format(len(lSet), len(uSet)))

    if cfg.TRAIN.DATASET.upper() == "IMAGENET":
        temp_cfg_worker = cfg.DATA_LOADER.NUM_WORKERS
        cfg.DATA_LOADER.NUM_WORKERS = 0

    if cfg.TRAIN.DATASET == "IMAGENET":
        dataObj = None
        noAugDataset = None

    elif cfg.TRAIN.DATASET == "STL10":
        oldmode = dataObj.eval_mode
        dataObj.eval_mode = True
        noAugDataset, _ = dataObj.getDatasetForVAAL(
            save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True
        )
        dataObj.eval_mode = oldmode

    else:
        oldmode = dataObj.eval_mode
        dataObj.eval_mode = True
        noAugDataset, _ = dataObj.getDataset(
            save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True
        )
        dataObj.eval_mode = oldmode

    # First train vae and disc
    vae, disc = train_vae_disc(cfg, lSet, uSet, noAugDataset, dataObj, debug)

    if cfg.TRAIN.DATASET == "IMAGENET":
        temp_vaal_bs = cfg.VAAL.VAE_BS
        cfg.VAAL.VAE_BS = cfg.TRAIN.BATCH_SIZE
        uSetLoader = imagenet_loader.construct_loader_no_aug(
            cfg, indices=uSet, isShuffle=False, isDistributed=False
        )  # , isVaalSampling=True)
        cfg.VAAL.VAE_BS = temp_vaal_bs
    else:
        uSetLoader = dataObj.getSequentialDataLoader(
            indexes=uSet,
            batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
            data=noAugDataset,
        )

    # Do active sampling
    print("Setting vae and disc in eval mode..!")
    vae.eval()
    disc.eval()
    print("Done!!")

    sampler = vs.AdversarySampler(budget=cfg.ACTIVE_LEARNING.BUDGET_SIZE)
    print("call vae sampling to get activeSet")
    activeSet, uSet = sampler.sample_for_labeling(
        vae=vae, discriminator=disc, unlabeled_dataloader=uSetLoader, uSet=uSet, cfg=cfg
    )

    lSet = np.append(lSet, activeSet)

    # save arrays in npy format
    save_numpy_arrays(
        [lSet, uSet, activeSet], ["lSet", "uSet", "activeSet"], cfg.OUT_DIR
    )
    # save arrays in txt format
    save_numpy_arrays(
        [lSet, uSet, activeSet],
        ["lSet", "uSet", "activeSet"],
        cfg.OUT_DIR,
        saveinText=True,
    )

    if cfg.TRAIN.DATASET.lower() == "imagenet":
        cfg.TRAIN.IM_SIZE = temp_old_im_size
        cfg.DATA_LOADER.NUM_WORKERS = temp_cfg_worker

    # Dump cfg file --
    temp_cfg = deepcopy(cfg)
    temp_cfg.ACTIVE_LEARNING.ACTIVATE = True
    temp_cfg.ACTIVE_LEARNING.LSET_PATH = os.path.join(temp_cfg.OUT_DIR, "lSet.npy")
    temp_cfg.ACTIVE_LEARNING.USET_PATH = os.path.join(temp_cfg.OUT_DIR, "uSet.npy")
    custom_dump_cfg(temp_cfg)


def vaal_active_sampling_minus_disc(cfg, dataObj, debug=False):

    lSet_path = cfg.ACTIVE_LEARNING.LSET_PATH
    uSet_path = cfg.ACTIVE_LEARNING.USET_PATH

    lSet = np.load(lSet_path, allow_pickle=True)
    uSet = np.load(uSet_path, allow_pickle=True)

    # trainDataset = dataObj.getDataset(save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True)
    if cfg.TRAIN.DATASET == "IMAGENET":
        dataObj = None
        noAugDataset = None
    else:
        oldmode = dataObj.eval_mode
        dataObj.eval_mode = True
        noAugDataset, _ = dataObj.getDataset(
            save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True
        )
        dataObj.eval_mode = oldmode

    # First train vae
    vae = train_vae(cfg, lSet, uSet, noAugDataset, dataObj, debug)

    if cfg.TRAIN.DATASET == "IMAGENET":
        lSetLoader = imagenet_loader.construct_loader_no_aug(
            cfg, indices=lSet, isShuffle=False, isDistributed=False
        )  # , isVaalSampling=True)
        uSetLoader = imagenet_loader.construct_loader_no_aug(
            cfg, indices=uSet, isShuffle=False, isDistributed=False
        )  # , isVaalSampling=True)
    else:

        lSetLoader = dataObj.getIndexesDataLoader(
            indexes=lSet,
            batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
            data=noAugDataset,
        )

        uSetLoader = dataObj.getSequentialDataLoader(
            indexes=uSet,
            batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
            data=noAugDataset,
        )

    # Do active sampling
    vae.eval()
    sampler = vs.AdversarySampler(budget=cfg.ACTIVE_LEARNING.BUDGET_SIZE)
    with torch.no_grad():
        activeSet, uSet = sampler.vae_sample_for_labeling(
            vae=vae,
            uSet=uSet,
            lSet=lSet,
            unlabeled_dataloader=uSetLoader,
            lSetLoader=lSetLoader,
        )

    lSet = np.append(lSet, activeSet)

    save_numpy_arrays(
        [lSet, uSet, activeSet], ["lSet", "uSet", "activeSet"], cfg.OUT_DIR
    )
    save_numpy_arrays(
        [lSet, uSet, activeSet],
        ["lSet", "uSet", "activeSet"],
        cfg.OUT_DIR,
        saveinText=True,
    )


tempArgsFile = sys.argv[1]

# Getting back the objects:
with open(tempArgsFile, "rb") as f:  # Python 3: open(..., 'rb')
    cfg, dataObj = pickle.load(f)

if cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal":
    # Run original vaal
    print("--------------------------")
    print("Running VAAL Sampling")
    print("--------------------------")
    print("dataObj: {}".format(dataObj))
    vaal_active_sampling(cfg, dataObj, debug=True)
elif cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal_minus_disc":
    # Run vaal[-d]
    print("--------------------------")
    print("Running VAAL MINUS DISC Sampling")
    print("--------------------------")
    vaal_active_sampling_minus_disc(cfg, dataObj, debug=True)
