import torch
import torch.nn as nn

import os
import sys
import argparse

import torchcontrib
import pickle

from glob import glob
import copy

from yacs.config import CfgNode
from al_utils.data import Data as custom_Data

from pycls.core.model_builder import build_model
from tqdm import tqdm
import numpy as np


def get_optimizer_model(
    cfg: CfgNode,
    model_path: str,
    active_sampling: bool = False,
    isDistributed: bool = False,
):
    """Loads the optimizer and model"""

    model = build_model(cfg, active_sampling, isDistributed)

    if cfg.OPTIM.TYPE.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.OPTIM.BASE_LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )
    elif cfg.OPTIM.TYPE.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.OPTIM.BASE_LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError

    print(f"Model loading from path: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # optimizer load_state_dict loads the parameters in cpu, however for training them we require in gpu
    current_id = torch.cuda.current_device()
    for state in optimizer.state.values():
        for key, val in state.items():
            # if isinstance(v, torch.Tensor):
            if torch.is_tensor(val):
                state[key] = val.cuda(current_id)

    print("Curent_id: {}".format(current_id))
    model = model.cuda(current_id)

    model = torch.nn.DataParallel(
        model, device_ids=[i for i in range(torch.cuda.device_count())]
    )
    return optimizer, model


def swa_train(cfg, args, optimizer, model, lSetLoader, bn_lSetLoader, valSetLoader):
    """Function implementing SWA postraining."""
    current_id = torch.cuda.current_device()

    print(
        f"SWA config- start={cfg.SWA_MODE.START_ITER}, freq={cfg.SWA_MODE.FREQ}, swa_lr={cfg.SWA_MODE.LR}"
    )
    swa_optimizer = torchcontrib.optim.SWA(
        optimizer,
        swa_start=cfg.SWA_MODE.START_ITER,
        swa_freq=cfg.SWA_MODE.FREQ,
        swa_lr=cfg.SWA_MODE.LR,
    )
    print(f"SWA Optimizer: {swa_optimizer}")

    print("Training SWA for {} epochs.".format(args.swa_epochs))
    temp_max_itrs = len(lSetLoader)
    print(f"len(lSetLoader): {len(lSetLoader)}")
    temp_cur_itr = 0

    model.train()

    loss = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.swa_epochs), desc="Training SWA"):
        temp_cur_itr = 0
        for x, y in lSetLoader:
            x = x.cuda(current_id)
            y = y.cuda(current_id)

            output = model(x)

            error = loss(output, y)

            swa_optimizer.zero_grad()
            error.backward()
            swa_optimizer.step()

            temp_cur_itr += 1

        # print("Epoch {} Done!! Train Loss: {}".format(epoch, error.item()))

    print("Averaging weights -- SWA")
    swa_optimizer.swap_swa_sgd()
    print("Done!!")

    print("Updating BN")

    swa_optimizer.bn_update(loader=lSetLoader, model=model)
    print("Done!!")

    # Check val accuracy
    print("Evaluating validation accuracy")
    model.eval()

    accuracy = get_validation_accuracy(valSetLoader, model, current_id)

    print("Validation Accuracy with SWA model: {}".format(accuracy))

    print("Saving SWA model")
    print("len(cfg.NUM_GPUS): {}".format(cfg.NUM_GPUS))
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # when model is on single gpu then model state_dict contains keyword module
    # So we will simply remove <module> from dictionary.
    isModuleStrPresent = False
    for k in sd.keys():
        if k.find("module.") == -1:
            continue
        isModuleStrPresent = True
        break

    if isModuleStrPresent:
        print("SWA checkpoint contains module present in keys.")
        print("So now removing 'module' strings")
        # remove module strings
        from collections import OrderedDict

        new_ckpt_dict = OrderedDict()
        for k, v in sd.items():
            tmp_key = k.replace("module.", "")
            new_ckpt_dict[tmp_key] = v

        sd = copy.deepcopy(new_ckpt_dict)
        print("Done!!")

    # sd = model.state_dict()

    save_epoch = cfg.OPTIM.MAX_EPOCH + 1
    # Record the state
    checkpoint = {
        "epoch": save_epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": None,
    }
    # Write the checkpoint
    save_path = os.path.join(cfg.OUT_DIR, "checkpoints") + "/"
    checkpoint_file = save_path + "[SWA]valSet_acc_{}_model_epoch_{:04}.pyth".format(
        accuracy, save_epoch
    )
    print("---Before SAVING SWA MODEL----")
    torch.save(checkpoint, checkpoint_file)
    print("SAVED SWA model")
    print("SWA Model saved at {}".format(checkpoint_file))


def get_validation_accuracy(valSetLoader, model, current_id):
    """Returns the validation accuracy."""
    totalDataPoints = 0.0
    totalCorrectClfPoints = 0.0
    for i, (x, y) in enumerate(tqdm(valSetLoader, desc="Validation Accuracy")):
        x = x.cuda(current_id)
        y = y.cuda(current_id)

        pred = model(x)

        _, prediction = torch.max(pred.data, dim=1)
        totalDataPoints += y.size(0)
        totalCorrectClfPoints += (prediction == y).sum().item()

    accuracy = totalCorrectClfPoints / totalDataPoints
    accuracy *= 100.0
    return accuracy


def run_swa_on_trial(trial_path, cfg, args):
    """Runs SWA post-training on given trial"""
    print(f"--- Working on {trial_path} ---")
    cfg_file_path = os.path.join(trial_path, "config.yaml")
    cfg.merge_from_file(cfg_file_path)

    # To ensure we control the port number for DDP
    # cfg.PORT = args.port

    print("=========== : ", glob(os.path.join(trial_path, "checkpoints", "*.pyth")))
    model_path = [f for f in glob(os.path.join(trial_path, "checkpoints", "*.pyth"))]

    if len(model_path) == 0:
        # if there is no model present
        print(f"No model present at {trial_path}")
        print("So skipping it !!")
        return
    # assert len(model_path) == 1 or len(model_path) == 2

    if len(model_path) == 2:
        print("SWA models are already present at ", model_path)
        return

    model_path = model_path[0]
    optimizer, model = get_optimizer_model(
        cfg, model_path
    )  # , cfg.ACTIVE_LEARNING.ACTIVATE)#, active_sampling, isDistributed)

    print("optimizer: ", optimizer)

    # paths
    lSetPath = cfg.ACTIVE_LEARNING.LSET_PATH
    uSetPath = cfg.ACTIVE_LEARNING.USET_PATH
    valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH

    dataObj = custom_Data(dataset=cfg.TRAIN.DATASET, israndAug=cfg.RANDAUG.ACTIVATE)
    dataObj.rand_augment_N = cfg.RANDAUG.N
    dataObj.rand_augment_M = cfg.RANDAUG.M

    lSet, _, valSet = dataObj.loadPartitions(
        lSetPath=lSetPath, uSetPath=uSetPath, valSetPath=valSetPath
    )

    trainDataset, n_TrainDatapts = dataObj.getDataset(
        save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True
    )
    # Because this happens after active learning process then
    if cfg.ACTIVE_LEARNING.ACTIVATE and cfg.ACTIVE_LEARNING.NOISY_ORACLE > 0.0:

        if cfg.TRAIN.DATASET == "IMAGENET":
            raise NotImplementedError

        print("============= ADDING NOISE [SWA]=============")
        noise_percent = cfg.ACTIVE_LEARNING.NOISY_ORACLE

        temp_activeset_fpath = os.path.abspath(
            os.path.join(cfg.ACTIVE_LEARNING.LSET_PATH, os.pardir)
        )
        temp_activeset_fpath = os.path.join(temp_activeset_fpath, "activeSet.npy")

        print(f"accessing activeset at {temp_activeset_fpath}")
        activeSet = np.load(temp_activeset_fpath, allow_pickle=True)

        noise_idx = np.arange(start=0, stop=len(activeSet))
        np.random.seed(
            1536
        )  # ensures that in every automl iteration you get the same random noise in training dataset.
        np.random.shuffle(noise_idx)
        noise_idx = noise_idx[0 : int(noise_percent * len(activeSet))]
        print("len(noise_idx): ", len(noise_idx))
        active_noise_idx = activeSet[noise_idx]
        for idx in active_noise_idx:
            trainDataset.targets[idx] = np.random.randint(0, cfg.MODEL.NUM_CLASSES, 1)[
                0
            ]

        print("=============== DONE ================")

    # To get reference to data which has no transformations applied
    oldmode = dataObj.eval_mode
    dataObj.eval_mode = True  # To remove any transforms

    print("==== Loading valDataset ====")
    valDataset, _ = dataObj.getDataset(
        save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True
    )

    print("==== Loading noAugDataset ====")
    noAugDataset, _ = dataObj.getDataset(
        save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True
    )

    dataObj.eval_mode = oldmode

    lSetLoader = dataObj.getIndexesDataLoader(
        indexes=lSet, batch_size=int(cfg.TRAIN.BATCH_SIZE), data=trainDataset
    )
    bn_lSetLoader = dataObj.getIndexesDataLoader(
        indexes=lSet, batch_size=int(cfg.TRAIN.BATCH_SIZE), data=noAugDataset
    )
    valSetLoader = dataObj.getIndexesDataLoader(
        indexes=valSet, batch_size=32, data=noAugDataset
    )

    swa_train(cfg, args, optimizer, model, lSetLoader, bn_lSetLoader, valSetLoader)


tempArgsFile = sys.argv[1]

# Getting back the objects:
with open(tempArgsFile, "rb") as f:  # Python 3: open(..., 'rb')
    cfg, args = pickle.load(f)

# necessary args
aml_models = cfg.OUT_DIR

all_trials_path = [f for f in glob(os.path.join(aml_models, "trial-*"))]
all_trials_path.sort()
num_trails = len(all_trials_path)

print("Number of trials: ", num_trails)

for trial_path in all_trials_path:
    run_swa_on_trial(trial_path, cfg, args)
