#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

# Modified by Prateek Munjal from official pycls codebase inorder to add the AL functionality
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""

import argparse
import numpy as np
import os
import optuna
import sys
import torch
import pickle
import subprocess as sp
import copy

from pycls.core.config import assert_cfg

from pycls.core.config import dump_cfg
from pycls.core.config import custom_dump_cfg
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.utils.meters import ValMeter

import pycls.core.losses as losses
import pycls.core.model_builder as model_builder
import pycls.core.optimizer as optim
import pycls.utils.benchmark as bu
import pycls.utils.checkpoint as cu
import pycls.utils.distributed as du
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.multiprocessing as mpu
import pycls.utils.net as nu

import pycls.datasets.loader as imagenet_loader

from helper.args_util import get_main_args
from helper.args_util import parse_args
from helper.args_util import get_al_args

from helper.subprocess_utils import vaal_sampling_util
from helper.subprocess_utils import active_sampling
from helper.subprocess_utils import test_net_subprocess_call
from helper.subprocess_utils import SWA_subprocess_call

from helper.path_extractor import get_latest_model_path
from helper.path_extractor import get_best_model_path
from helper.path_extractor import update_lset_uset_paths

logger = lu.get_logger(__name__)

plot_epoch_xvalues = []
plot_epoch_yvalues = []
plot_it_xvalues = []
plot_it_y_values = []


def plot_arrays(cfg, x_vals, y_vals, x_name, y_name, dataset_name, isDebug=False):
    """Basic utility to plot X vs Y line graphs.

    Args:
        cfg: Reference to the config yaml
        x_vals: values on x-axis
        y_vals: values on y-axis
        x_name: Label on x-axis
        y_name: Label on y-axis
        dataset_name: Dataset name.
        isDebug (bool, optional): Switch for debug mode. Defaults to False.
    """
    if not du.is_master_proc(cfg):
        return

    import matplotlib.pyplot as plt

    temp_name = "{}_vs_{}".format(x_name, y_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title("Dataset: {}; {}".format(dataset_name, temp_name))
    plt.plot(x_vals, y_vals)

    if isDebug:
        print(f"plot_saved at {cfg.OUT_DIR+temp_name}.png")

    if cfg.TRAIN.TRANSFER_EXP:
        temp_path = (
            os.path.join(
                "transfer_experiment",
                cfg.MODEL.TRANSFER_MODEL_TYPE
                + "_depth_"
                + str(cfg.MODEL.TRANSFER_MODEL_DEPTH),
            )
            + "/"
        )
        plt.savefig(cfg.OUT_DIR + temp_path + temp_name + ".png")
    plt.savefig(cfg.OUT_DIR + temp_name + ".png")
    plt.close()


def save_plot_values(
    cfg, temp_arrays, temp_names, isParallel=True, saveInTextFormat=False, isDebug=True
):

    """Saves arrays provided in the list in npy format"""
    # return if not master process
    if isParallel:
        if not du.is_master_proc(cfg):
            return

    for i in range(len(temp_arrays)):
        temp_arrays[i] = np.array(temp_arrays[i])
        temp_dir = cfg.OUT_DIR
        if cfg.TRAIN.TRANSFER_EXP:
            temp_dir += (
                os.path.join(
                    "transfer_experiment",
                    cfg.MODEL.TRANSFER_MODEL_TYPE
                    + "_depth_"
                    + str(cfg.MODEL.TRANSFER_MODEL_DEPTH),
                )
                + "/"
            )

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if saveInTextFormat:
            if isDebug:
                print(
                    f"Saving {temp_names[i]} at {temp_dir+temp_names[i]}.txt in text format!!"
                )
            np.savetxt(temp_dir + temp_names[i] + ".txt", temp_arrays[i], fmt="%d")
        else:
            if isDebug:
                print(
                    f"Saving {temp_names[i]} at {temp_dir+temp_names[i]}.npy in numpy format!!"
                )
            np.save(temp_dir + temp_names[i] + ".npy", temp_arrays[i])


def is_eval_epoch(cfg, cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or (
        cur_epoch + 1
    ) == cfg.OPTIM.MAX_EPOCH


def log_model_info(model):
    """Logs model info"""
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(mu.params_count(model)))
    logger.info("Flops: {:,}".format(mu.flops_count(model)))


def train_epoch(
    train_loader,
    model,
    loss_fun,
    optimizer,
    train_meter,
    cur_epoch,
    cfg,
    clf_iter_count,
    clf_change_lr_iter,
    clf_max_iter,
):
    """Performs one epoch of training."""

    if cfg.NUM_GPUS > 1:
        train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    # Enable training mode
    model.train()
    train_meter.iter_tic()  # This basically notes the start time in timer class defined in utils/timer.py

    len_train_loader = len(train_loader)
    for cur_iter, (inputs, labels) in enumerate(train_loader):

        # ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)

        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)

        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the parametersSWA
        optimizer.step()

        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])

        # Combine the stats across the GPUs
        if cfg.NUM_GPUS > 1:
            # Average error and losses across GPUs
            # Also this this calls wait method on reductions so we are ensured
            # to obtain synchronized results
            loss, top1_err = du.scaled_all_reduce(cfg, [loss, top1_err])

        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()

        # #ONLY MASTER PROCESS SHOULD WRITE TO TENSORBOARD
        if du.is_master_proc(cfg):
            if cur_iter is not 0 and cur_iter % 5 == 0:
                # because cur_epoch starts with 0
                plot_it_xvalues.append((cur_epoch) * len_train_loader + cur_iter)
                plot_it_y_values.append(loss)

                save_plot_values(
                    cfg,
                    [plot_it_xvalues, plot_it_y_values],
                    ["plot_it_xvalues.npy", "plot_it_y_values.npy"],
                    isDebug=False,
                )
                plot_arrays(
                    cfg,
                    x_vals=plot_it_xvalues,
                    y_vals=plot_it_y_values,
                    x_name="Iterations",
                    y_name="Loss",
                    dataset_name=cfg.TRAIN.DATASET,
                )

        # Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()

        train_meter.update_stats(
            top1_err=top1_err, loss=loss, lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS
        )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return loss, clf_iter_count


@torch.no_grad()
def test_epoch(cfg, test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.0
    totalSamples = 0.0
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        inputs = inputs.type(torch.cuda.FloatTensor)

        # Compute the predictions
        preds = model(inputs)

        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])

        # Combine the errors across the GPUs
        if cfg.NUM_GPUS > 1:
            top1_err = du.scaled_all_reduce(cfg, [top1_err])
            # as above returns a list
            top1_err = top1_err[0]

        # Copy the errors from GPU to CPU (sync point)
        top1_err = top1_err.item()

        # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
        misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
        totalSamples += inputs.size(0) * cfg.NUM_GPUS

        test_meter.iter_toc()

        # Update and log stats
        test_meter.update_stats(
            top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
        )
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()

    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    return misclassifications / totalSamples


def train_model(
    best_val_acc,
    best_val_epoch,
    trainDataset,
    valDataset,
    dataObj,
    cfg,
    trial,
    isPruning,
):
    """Trains the model."""

    global plot_epoch_xvalues
    global plot_epoch_yvalues
    global plot_it_xvalues
    global plot_it_y_values

    plot_epoch_xvalues = []
    plot_epoch_yvalues = []
    plot_it_xvalues = []
    plot_it_y_values = []

    # Build the model (before the loaders to speed up debugging)
    model = model_builder.build_model(
        cfg, active_sampling=cfg.ACTIVE_LEARNING.ACTIVATE, isDistributed=True
    )

    # Define the loss function
    if cfg.TRAIN.IMBALANCED:
        if cfg.TRAIN.DATASET == "IMAGENET":
            raise NotImplementedError
        temp_lSet, _, _ = dataObj.loadPartitions(
            lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH,
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH,
            valSetPath=cfg.ACTIVE_LEARNING.VALSET_PATH,
        )
        temp_weights = dataObj.getClassWeightsFromDataset(
            dataset=trainDataset, index_set=temp_lSet, bs=cfg.TRAIN.BATCH_SIZE
        )
        # print(f"temp_weights: {temp_weights}")
        loss_fun = torch.nn.CrossEntropyLoss(
            weight=temp_weights.cuda(torch.cuda.current_device())
        )
        print("Weighted cross entropy loss chosen as loss function")
        print(
            "Sum of weights: {} and weights.shape: {}".format(
                torch.sum(temp_weights), temp_weights.shape
            )
        )
    else:
        loss_fun = losses.get_loss_fun()

    # Construct the optimizer
    optimizer = optim.construct_optimizer(cfg, model)

    print("========OPTIMIZER========")
    print("optimizer: {}".format(optimizer))
    print("=========================")

    start_epoch = 0

    # Load initial weights if there are any
    if cfg.TRAIN.WEIGHTS:
        start_epoch = cu.load_checkpoint(cfg, cfg.TRAIN.WEIGHTS, model, optimizer)
        logger.info("=================================")
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
        logger.info("Base LR: {}".format(cfg.OPTIM.BASE_LR))
        logger.info("=================================")

    # If active learning mode then there has to be some starting point model
    if cfg.ACTIVE_LEARNING.ACTIVATE:

        if cfg.TRAIN.DATASET in ["CIFAR10", "CIFAR100", "SVHN", "MNIST", "STL10"]:
            print("==================================")
            print(
                "We are not finetuning over the provided dataset {}".format(
                    cfg.TRAIN.DATASET
                )
            )
            print(
                "So Although we can load best model from path: {} -- but we won't do on CIFAR datsets".format(
                    cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR
                )
            )
            print("Exiting model loafing function")
            print("==================================")

        else:
            cu.load_checkpoint(cfg, cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR, model)
            logger.info("=================================")
            logger.info(
                "Loaded initial weights from: {}".format(
                    cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR
                )
            )
            logger.info("Base LR: {}".format(cfg.OPTIM.BASE_LR))
            logger.info("=================================")

    # check if randAug activated
    if cfg.RANDAUG.ACTIVATE:
        print("==========================================")
        print(
            "RandAug activated with N: {} and M: {}".format(
                cfg.RANDAUG.N, cfg.RANDAUG.M
            )
        )
        print("==========================================")

    # Compute precise time
    if start_epoch == 0 and cfg.PREC_TIME.ENABLED:
        logger.info("Computing precise time...")
        bu.compute_precise_time(model, loss_fun)
        nu.reset_bn_stats(model)

    # Create data loaders

    lSet = []
    uSet = []

    # handles when we pass cifar/svhn datasets
    if cfg.TRAIN.DATASET in ["CIFAR10", "CIFAR100", "SVHN", "MNIST", "STL10", "RSNA"]:
        # get partitions
        lSet, uSet, valSet = dataObj.loadPartitions(
            lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH,
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH,
            valSetPath=cfg.ACTIVE_LEARNING.VALSET_PATH,
        )
        print("====== Partitions Loaded =======")
        print("lSet: {}, uSet:{}, valSet: {}".format(len(lSet), len(uSet), len(valSet)))
        print("================================")

        train_loader = dataObj.getDistributedIndexesDataLoader(
            cfg=cfg,
            indexes=lSet,
            batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
            data=trainDataset,
            n_worker=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=True,
        )

        valSetLoader = dataObj.getDistributedIndexesDataLoader(
            cfg=cfg,
            indexes=valSet,
            batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
            data=valDataset,
            n_worker=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
            allowRepeat=False,
        )

        # Loading test partition
        logger.info("==== Loading TestDataset ====")
        oldmode = dataObj.eval_mode
        dataObj.eval_mode = True
        testDataset, n_TestDatapts = dataObj.getDataset(
            save_dir=cfg.TEST_DIR, isTrain=False, isDownload=True
        )
        print("Number of testing datapoints: {}".format(n_TestDatapts))

        test_loader = dataObj.getDistributedIndexesDataLoader(
            cfg=cfg,
            indexes=None,
            batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
            data=testDataset,
            n_worker=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
            allowRepeat=False,
        )
        dataObj.eval_mode = oldmode

    elif cfg.TRAIN.DATASET == "IMAGENET":
        logger.info("==========================")
        logger.info("Trying to load imagenet dataset")
        logger.info("==========================")

        train_loader, valSetLoader = imagenet_loader.get_data_loaders(cfg)
        test_loader = imagenet_loader.construct_test_loader(cfg)
    else:
        logger.info(f"Dataset {cfg.TRAIN.DATASET} currently not supported")
        raise NotImplementedError

    # Create meters
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(valSetLoader), cfg)
    test_meter = TestMeter(len(test_loader), cfg)

    # Perform the training loop
    print("Len(train_loader): {}".format(len(train_loader)))
    logger.info("Start epoch: {}".format(start_epoch + 1))
    val_set_acc = 0.0

    temp_best_val_acc = 0.0
    temp_best_val_epoch = 0

    ##best checkpoint states
    best_model_state = None
    best_opt_state = None

    val_acc_epochs_x = []
    val_acc_epochs_y = []

    clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(lSet) / cfg.TRAIN.BATCH_SIZE)
    clf_change_lr_iter = clf_train_iterations // 25

    clf_iter_count = 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # # Train for one epoch
        train_loss, clf_iter_count = train_epoch(
            train_loader,
            model,
            loss_fun,
            optimizer,
            train_meter,
            cur_epoch,
            cfg,
            clf_iter_count,
            clf_change_lr_iter,
            clf_train_iterations,
        )

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)

        # # Evaluate the model
        if is_eval_epoch(cfg, cur_epoch):
            # Original code passes on testLoader but we want to compute on val Set
            val_set_err = test_epoch(cfg, valSetLoader, model, val_meter, cur_epoch)
            val_set_acc = 100.0 - val_set_err

            if temp_best_val_acc < val_set_acc:
                temp_best_val_acc = val_set_acc
                temp_best_val_epoch = cur_epoch + 1

                # Save best model and optimizer state for checkpointing
                model.eval()

                best_model_state = (
                    model.module.state_dict()
                    if cfg.NUM_GPUS > 1
                    else model.state_dict()
                )
                best_opt_state = optimizer.state_dict()
                model.train()

            # log if master process
            if du.is_master_proc(cfg):
                # as we start from 0 epoch
                val_acc_epochs_x.append(cur_epoch + 1)
                val_acc_epochs_y.append(val_set_acc)

        #######################
        # Save a checkpoint
        ######################
        if cfg.TRAIN.DATASET == "IMAGENET" and cu.is_checkpoint_epoch(cfg, cur_epoch):
            # named_save_checkpoint saves model with cur_epoch+1 in name
            checkpoint_file = cu.named_save_checkpoint(
                cfg, "valSet_acc_" + str(val_set_acc), model, optimizer, cur_epoch
            )
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))

        # ##Tensorboard for loss vs epoch
        if du.is_master_proc(cfg):
            plot_epoch_xvalues.append(cur_epoch)
            plot_epoch_yvalues.append(train_loss)

            save_plot_values(
                cfg,
                [
                    plot_epoch_xvalues,
                    plot_epoch_yvalues,
                    plot_it_xvalues,
                    plot_it_y_values,
                    val_acc_epochs_x,
                    val_acc_epochs_y,
                ],
                [
                    "plot_epoch_xvalues.npy",
                    "plot_epoch_yvalues.npy",
                    "plot_it_xvalues.npy",
                    "plot_it_y_values.npy",
                    "val_acc_epochs_x",
                    "val_acc_epochs_y",
                ],
                isDebug=False,
            )
            logger.info("Successfully logged numpy arrays!!")

            ##PLOT arrays
            plot_arrays(
                cfg,
                x_vals=plot_epoch_xvalues,
                y_vals=plot_epoch_yvalues,
                x_name="Epochs",
                y_name="Loss",
                dataset_name=cfg.TRAIN.DATASET,
            )

            plot_arrays(
                cfg,
                x_vals=val_acc_epochs_x,
                y_vals=val_acc_epochs_y,
                x_name="Epochs",
                y_name="Validation accuracy",
                dataset_name=cfg.TRAIN.DATASET,
            )

            print("~~~ isPruning Flag: ", isPruning)
            print("~~~ isEvalEpoch: ", is_eval_epoch(cfg, cur_epoch))

            if (
                isPruning
                and cur_epoch != 0
                and (cur_epoch % 20 == 0)
                and is_eval_epoch(cfg, cur_epoch)
            ):

                print("======================================\n")
                print("Inside pruning: -- ", isPruning)
                print("======================================\n")

                trial.report(val_set_acc, cur_epoch)

                if trial.should_prune():
                    print("======================================\n")
                    print("Getting pruned!!")
                    print("======================================\n")
                    raise optuna.exceptions.TrialPruned()

    save_plot_values(
        cfg,
        [
            plot_epoch_xvalues,
            plot_epoch_yvalues,
            plot_it_xvalues,
            plot_it_y_values,
            val_acc_epochs_x,
            val_acc_epochs_y,
        ],
        [
            "plot_epoch_xvalues.npy",
            "plot_epoch_yvalues.npy",
            "plot_it_xvalues.npy",
            "plot_it_y_values.npy",
            "val_acc_epochs_x",
            "val_acc_epochs_y",
        ],
    )

    if du.is_master_proc(cfg):
        # update shared variable -- iff process is master process
        # if distributed training
        if cfg.NUM_GPUS > 1:
            best_val_acc.value = temp_best_val_acc
            best_val_epoch.value = temp_best_val_epoch
        else:
            best_val_acc = temp_best_val_acc
            best_val_epoch = temp_best_val_epoch

        """
        SAVES the best model checkpoint
        """

        checkpoint_file = cu.state_save_checkpoint(
            cfg=cfg,
            info="vlBest_acc_" + str(temp_best_val_acc),
            model_state=best_model_state,
            optimizer_state=best_opt_state,
            epoch=temp_best_val_epoch,
        )

        logger.info("Wrote checkpoint to: {}".format(checkpoint_file))

    if not cfg.NUM_GPUS > 1:
        return best_val_acc, best_val_epoch


def single_proc_train(
    val_acc, val_epoch, trainDataset, valDataset, dataObj, cfg, trial, isPruning
):
    """Performs single process training."""

    # Setup logging
    lu.setup_logging(cfg)

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Train the model
    if cfg.NUM_GPUS > 1:
        train_model(
            val_acc, val_epoch, trainDataset, valDataset, dataObj, cfg, trial, isPruning
        )
    else:
        return train_model(
            val_acc, val_epoch, trainDataset, valDataset, dataObj, cfg, trial, isPruning
        )


def ensemble_sampling(
    args,
    cfg,
    main_args,
    temp_out_dir,
    trainDataset,
    valDataset,
    noAugDataset,
    dataObj,
    debug=True,
):

    temp_cfg = copy.deepcopy(cfg)

    if debug:
        logger.info("Inside Ensemble sampling function")

    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(main_args)

    num_ensembles = args.num_ensembles

    ENS_DIR_SUFFIX = "ens_model_"

    current_device = 0

    # train num_ensemble models
    print("==========================")
    print(f"Num_Ensembles: {num_ensembles}")
    print(f"main_args: {main_args}")
    print(f"initial temp_out_dir: {temp_out_dir}")
    print(f"cfg.ACTIVE_LEARNING.ACTIVATE: {cfg.ACTIVE_LEARNING.ACTIVATE}")
    print(f"cfg.ACTIVE_LEARNING.LSET_PATH: {cfg.ACTIVE_LEARNING.LSET_PATH}")
    print(f"cfg.ACTIVE_LEARNING.USET_PATH: {cfg.ACTIVE_LEARNING.USET_PATH}")
    print(f"cfg.ACTIVE_LEARNING.VALSET_PATH: {cfg.ACTIVE_LEARNING.VALSET_PATH}")
    print(f"cfg.ACTIVE_LEARNING.SAMPLING_FN: {cfg.ACTIVE_LEARNING.SAMPLING_FN}")
    print("==========================")

    model_paths = []
    for i in range(num_ensembles):
        print("=== Training ensemble [{}/{}] ===".format(i + 1, num_ensembles))
        cfg.defrost()  # to make cfg mutable
        """
        Switch off any regularization if there is any
        """

        print(f"Rand_Aug was switched to {cfg.RANDAUG.ACTIVATE}")
        if cfg.RANDAUG.ACTIVATE:
            cfg.RANDAUG.ACTIVATE = False
            print(f"Setting RandAug to --> {cfg.RANDAUG.ACTIVATE}")

        print(f"SWA was switched to {cfg.SWA_MODE.ACTIVATE}")
        if cfg.SWA_MODE.ACTIVATE:
            cfg.SWA_MODE.ACTIVATE = False
            print(f"Setting SWA MODE to --> {cfg.SWA_MODE.ACTIVATE}")

        cfg.OPTIM.MAX_EPOCH = args.ens_epochs
        print(f"Max epochs for training ensemble: {cfg.OPTIM.MAX_EPOCH}")

        cfg.RNG_SEED += i
        cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget_size
        cfg.TEST.BATCH_SIZE = args.test_batch_size
        cfg.TEST.DATASET = args.dataset
        cfg.TRAIN.BATCH_SIZE = args.train_batch_size
        cfg.TRAIN.DATASET = args.dataset
        cfg.TRAIN.EVAL_PERIOD = args.eval_period
        cfg.TRAIN.CHECKPOINT_PERIOD = args.checkpoint_period
        cfg.TRAIN.IMBALANCED = args.isimbalanced
        cfg.ENSEMBLE.NUM_MODELS = num_ensembles
        cfg.ENSEMBLE.MODEL_TYPE = [str(cfg.MODEL.TYPE)]

        print(f"====== Ensemble OPTIM LR: {cfg.OPTIM.BASE_LR}=====")

        print("=== SEED: {} ===".format(cfg.RNG_SEED))
        cfg.OUT_DIR = temp_out_dir + ENS_DIR_SUFFIX + str(i + 1) + "/"

        model_paths.append(cfg.OUT_DIR)
        print(f"cfg.OUT_DIR: {cfg.OUT_DIR}")
        print(f"cfg.ACTIVE_LEARNING.BUDGET_SIZE: {cfg.ACTIVE_LEARNING.BUDGET_SIZE}")

        if os.path.exists(cfg.OUT_DIR):
            print(
                f"Skipping ensemble {i+1} learning as it already exists: {cfg.OUT_DIR}"
            )
        else:
            al_main(cfg, args, trainDataset, valDataset, dataObj, None, isSkipCfg=True)

    cfg.defrost()

    if debug:
        print(f"[Before] model_paths: {model_paths}")

    model_paths = [
        get_best_model_path(None, [], 0, "", False, directPath=md_path)
        for md_path in model_paths
    ]

    if debug:
        print(f"[After] model_paths: {model_paths}")

    temp_args = [model_paths, num_ensembles, noAugDataset, dataObj, temp_out_dir]
    active_sampling(cfg, ensemble_args=temp_args, debug=False)

    # Get original CFG back
    cfg = copy.deepcopy(temp_cfg)

    return 0


# this calls distributed training
def al_main(
    cfg, args, trainDataset, valDataset, dataObj, al_args=None, isSkipCfg=False
):

    """Main function running AL cycles"""

    if not isSkipCfg:
        # Load config options
        cfg.merge_from_file(args.cfg_file)

        if al_args is not None:
            cfg.merge_from_list(al_args)

    assert_cfg()
    cfg.freeze()

    # Ensure that the output dir exists
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Save the config
    dump_cfg(cfg)

    # Perform training
    if cfg.NUM_GPUS > 1:
        print("============================")
        print("Number of Gpus available for multiprocessing: {}".format(cfg.NUM_GPUS))
        print("============================")
        best_val_acc, best_val_epoch = mpu.multi_proc_run(
            num_proc=cfg.NUM_GPUS,
            fun=single_proc_train,
            fun_args=(trainDataset, valDataset, dataObj, cfg, 0, True),
        )
    else:
        temp_val_acc = 0.0
        temp_val_epoch = 0
        # val_acc, val_epoch, trainDataset, valDataset, dataObj, cfg
        best_val_acc, best_val_epoch = single_proc_train(
            temp_val_acc,
            temp_val_epoch,
            trainDataset,
            valDataset,
            dataObj,
            cfg,
            0,
            True,
        )

    cfg.defrost()  # Make cfg mutable for other operations
    return best_val_acc, best_val_epoch


def main(cfg):

    # Parse cmd line args
    args = parse_args()

    best_val_accuracies = []
    test_accuracies = []
    test_model_paths = []  # For verification purposes
    best_val_epochs = []
    temp_model_path = ""

    al_model_phase = args.al_mode
    print("== al_model_phase: {} ==".format(al_model_phase))
    al_start = args.init_partition
    sampling_fn = args.sampling_fn if al_model_phase else None

    dataset_name = args.dataset

    if al_model_phase:
        al_step = args.step_partition
        al_stop = al_start + args.al_max_iter * al_step
        data_splits = [round(i, 1) for i in np.arange(al_start, al_stop, al_step)]
    else:
        data_splits = [args.init_partition]

    al_max_iter = len(data_splits)
    i_start = 1 if al_max_iter > 1 else 0

    # compulsory arguments needed irrespective of active learning or not
    main_args = get_main_args(args)
    temp_out_dir = ""

    directory_specific = "vanilla"
    if args.isTransferExp:
        print(
            f"========= [Running Transfer Experiment; DIRECTORY SPECIFIC SET TO {args.transfer_dir_specific}] ========="
        )
        directory_specific = args.transfer_dir_specific
    else:

        if args.swa_mode and args.rand_aug:
            directory_specific = "swa_rand_aug"
        elif args.swa_mode:
            directory_specific = "swa"
        elif args.rand_aug:
            directory_specific = "rand_aug"
        else:
            print("========= [NO ADVANCED REGULARIZATION TRICK ACTIVATED] =========")

    print(f"Directory_specific: {directory_specific}")

    # ONLY SWA MODE

    # Construct datasets
    from al_utils.data import Data as custom_Data

    if args.dataset in ["CIFAR10", "CIFAR100", "SVHN", "MNIST", "STL10"]:
        dataObj = custom_Data(dataset=args.dataset, israndAug=args.rand_aug, args=args)
        logger.info("==== Loading trainDataset ====")
        trainDataset, n_TrainDatapts = dataObj.getDataset(
            save_dir=args.train_dir, isTrain=True, isDownload=True
        )

        # To get reference to data which has no transformations applied
        oldmode = dataObj.eval_mode
        dataObj.eval_mode = True  # To remove any transforms
        logger.info("==== Loading valDataset ====")
        valDataset, _ = dataObj.getDataset(
            save_dir=args.train_dir, isTrain=True, isDownload=True
        )
        logger.info("==== Loading noAugDataset ====")
        noAugDataset, _ = dataObj.getDataset(
            save_dir=args.train_dir, isTrain=True, isDownload=True
        )
        dataObj.eval_mode = oldmode
    elif args.dataset == "IMAGENET":
        trainDataset = None
        valDataset = None
        noAugDataset = None
        dataObj = None
        # All these are defined later as they need final values of cfg and yet cfg is not properly set
        pass
    else:
        logger.info(f"{args.dataset} dataset not handled yet.")
        raise NotImplementedError

    if args.only_swa:
        # USAGE: When we only want to run SWA on some model weights
        cfg.RANDAUG.ACTIVATE = args.rand_aug
        cfg.MODEL.DEPTH = args.model_depth
        cfg.MODEL.TYPE = args.model_type
        cfg.TRAIN.DATASET = args.dataset
        cfg.TRAIN.BATCH_SIZE = args.train_batch_size
        cfg.TEST.BATCH_SIZE = args.test_batch_size

        # To reflect our cmd arguments and config file changes in cfg
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(main_args)

        cfg.ACTIVE_LEARNING.LSET_PATH = args.lSetPath
        cfg.ACTIVE_LEARNING.USET_PATH = args.uSetPath
        cfg.ACTIVE_LEARNING.VALSET_PATH = args.valSetPath

        temp_out_dir = (
            args.out_dir
            + dataset_name
            + "/"
            + str(args.init_partition)
            + "/"
            + args.model_type
            + "_depth_"
            + str(args.model_depth)
            + "/"
            + directory_specific
            + "/"
        )
        logger.info(f"Temp_out_dir: {temp_out_dir}")
        if args.only_swa_partition == args.init_partition:
            temp_l_SetPath = args.lSetPath
            temp_u_SetPath = args.uSetPath
        else:
            temp_l_SetPath = (
                args.out_dir
                + args.dataset
                + "/"
                + str(args.only_swa_partition - args.step_partition)
                + "/"
                + args.model_type
                + "_depth_"
                + str(args.model_depth)
                + "/"
                + directory_specific
                + "/"
                + args.sampling_fn
                + "/lSet.npy"
            )
            temp_u_SetPath = (
                args.out_dir
                + args.dataset
                + "/"
                + str(args.only_swa_partition - args.step_partition)
                + "/"
                + args.model_type
                + "_depth_"
                + str(args.model_depth)
                + "/"
                + directory_specific
                + "/"
                + args.sampling_fn
                + "/uSet.npy"
            )

        latest_model_path = get_latest_model_path(
            dir_path=temp_out_dir + "checkpoints/"
        )

        print("temp_out_dir: {}".format(temp_out_dir))
        print("lsetPath: {}".format(temp_l_SetPath))
        print("uSetPath: {}".format(temp_u_SetPath))
        print("valSetPath: {}".format(args.valSetPath))
        print("latest_model_path: {}".format(latest_model_path))

        args.device_ids = np.arange(cfg.NUM_GPUS)

        argListSWA = [
            args,
            latest_model_path,
            temp_l_SetPath,
            temp_u_SetPath,
            temp_out_dir + "checkpoints/",
            trainDataset,
            noAugDataset,
            cfg,
        ]

        SWA_subprocess_call(argListSWA, debug=True)
        return

    # SWA will be called here if applied
    for i in range(i_start, al_max_iter):
        if al_model_phase:
            # Hierarchy followed -- [al_results/partition_size/dataset_name/model_type/directory_specific/sampling_fn/data_splits]

            if data_splits[i] == round(args.init_partition + args.step_partition, 1):
                # First time active learning
                al_args, temp_out_dir = get_al_args(
                    args, data_splits, i, directory_specific, alStart=True
                )

            else:
                al_args, temp_out_dir = get_al_args(
                    args, data_splits, i, directory_specific, alStart=False
                )

            cfg.merge_from_file(args.cfg_file)
            cfg.merge_from_list(main_args + al_args)

            assert_cfg()

            # Should we do active sampling or not?
            # If lSet, uSet and activeSet found in their target directories
            # then we skip sampling part for that particular iteration
            skip_sampling = True
            check_path = (
                args.out_dir
                + args.dataset
                + "/"
                + str(data_splits[i])
                + "/"
                + str(args.seed_id)
                + "/"
                + args.model_type
                + "_depth_"
                + str(args.model_depth)
                + "/"
                + directory_specific
                + "/"
                + args.sampling_fn
                + "/"
            )
            print("==============================")
            print(f"check_path: {check_path}")
            print("==============================")
            req_fnames = ["lSet", "uSet", "activeSet"]
            for fname in req_fnames:
                if os.path.exists(check_path + fname + ".npy") and os.path.exists(
                    check_path + fname + ".txt"
                ):
                    continue
                else:
                    skip_sampling = False
                    break

            if not skip_sampling:
                # do active sampling
                if cfg.ACTIVE_LEARNING.SAMPLING_FN in ["vaal", "vaal_minus_disc"]:
                    temp_old_im_size = cfg.TRAIN.IM_SIZE
                    if cfg.TRAIN.DATASET == "IMAGENET":
                        cfg.TRAIN.IM_SIZE = args.vaal_im_size

                    vaal_sampling_util(cfg, dataObj, debug=True)

                    if cfg.TRAIN.DATASET == "IMAGENET":
                        cfg.TRAIN.IM_SIZE = temp_old_im_size

                elif cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("ensemble"):
                    ensemble_sampling(
                        args,
                        cfg,
                        main_args,
                        temp_out_dir,
                        trainDataset,
                        valDataset,
                        noAugDataset,
                        dataObj,
                        debug=True,
                    )
                else:
                    active_sampling(cfg, debug=True)
            else:
                print(
                    "Sampling Skipped as index sets exists at path: {}".format(
                        check_path
                    )
                )

            # update lSetPath, uSetPath
            al_args = update_lset_uset_paths(
                al_args,
                args.out_dir
                + args.dataset
                + "/"
                + str(data_splits[i])
                + "/"
                + str(args.seed_id)
                + "/"
                + args.model_type
                + "_depth_"
                + str(args.model_depth)
                + "/"
                + directory_specific
                + "/"
                + args.sampling_fn
                + "/lSet.npy",
                args.out_dir
                + args.dataset
                + "/"
                + str(data_splits[i])
                + "/"
                + str(args.seed_id)
                + "/"
                + args.model_type
                + "_depth_"
                + str(args.model_depth)
                + "/"
                + directory_specific
                + "/"
                + args.sampling_fn
                + "/uSet.npy",
            )

        else:
            # base classifier phase
            temp_out_dir = (
                args.out_dir
                + dataset_name
                + "/"
                + str(data_splits[i])
                + "/"
                + str(args.seed_id)
                + "/"
                + args.model_type
                + "_depth_"
                + str(args.model_depth)
                + "/"
                + directory_specific
                + "/"
            )
            al_args = [
                "ACTIVE_LEARNING.LSET_PATH",
                args.lSetPath,
                "ACTIVE_LEARNING.USET_PATH",
                args.uSetPath,
                "OUT_DIR",
                temp_out_dir,
                "ACTIVE_LEARNING.VALSET_PATH",
                args.valSetPath,
                "ACTIVE_LEARNING.ACTIVATE",
                args.al_mode,
                "ACTIVE_LEARNING.DATA_SPLIT",
                args.init_partition,
                "DIR_SPECIFIC",
                directory_specific,
            ]

        # Make out_directory for saving results later
        os.makedirs(temp_out_dir, exist_ok=True)
        temp_al_args = al_args
        al_args = main_args + al_args

        print("========[CMD ARGUMNETS]=======")
        print("al_args: {}".format(al_args))
        print("Using data_splits: {}".format(data_splits))
        print("==============================")

        print("============================")
        print("Running AL iteration #{}".format(i))

        print("~~~~temp_out_dir: {}".format(temp_out_dir))
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("ensemble"):
            cfg.OUT_DIR = temp_out_dir

        # Because this happens after active learning process then
        if cfg.ACTIVE_LEARNING.ACTIVATE and cfg.ACTIVE_LEARNING.NOISY_ORACLE > 0.0:

            if cfg.TRAIN.DATASET == "IMAGENET":
                raise NotImplementedError

            print("============= ADDING NOISE =============")
            noise_percent = cfg.ACTIVE_LEARNING.NOISY_ORACLE
            # temp_data_split = cfg.ACTIVE_LEARNING.DATA_SPLIT
            activeSet = np.load(
                os.path.join(cfg.OUT_DIR, "activeSet.npy"), allow_pickle=True
            )
            noise_idx = np.arange(start=0, stop=len(activeSet))
            np.random.shuffle(noise_idx)
            noise_idx = noise_idx[0 : int(noise_percent * len(activeSet))]
            print("len(noise_idx): ", len(noise_idx))
            active_noise_idx = activeSet[noise_idx]
            for idx in active_noise_idx:
                trainDataset.targets[idx] = np.random.randint(
                    0, cfg.MODEL.NUM_CLASSES, 1
                )[0]

            print("=============== DONE ================")

        if cfg.TRAIN.TRANSFER_EXP == False and os.path.exists(
            temp_out_dir + "checkpoints/"
        ):
            print(
                f"=== Skipped Learning as path: [{temp_out_dir}checkpoints/] exists...==="
            )
            best_val_acc = 0
            best_val_epoch = 0
            print("temp_out_dir: ", temp_out_dir)
            cfg.merge_from_file(os.path.join(temp_out_dir, "config.yaml"))
            cfg.PORT = args.port

        else:
            best_val_acc, best_val_epoch = al_main(
                cfg, args, trainDataset, valDataset, dataObj, al_args
            )

        if cfg.TRAIN.TRANSFER_EXP:
            temp_out_dir = cu.get_checkpoint_dir_wo_checkpoint(cfg) + "/"
            print("temp_out_dir: {}".format(temp_out_dir))
            latest_model_path = get_latest_model_path(dir_path=temp_out_dir)
        else:
            latest_model_path = get_latest_model_path(
                dir_path=temp_out_dir + "checkpoints/"
            )

        print("temp_out_dir: {}".format(temp_out_dir))
        if best_val_epoch == 0 and best_val_acc == 0:
            model_info = os.path.split(latest_model_path)[1]
            best_val_epoch = int(model_info.split("_")[-1].split(".")[0])
            best_val_acc = float(model_info.split("_")[2])

        print("latest_model_path: {}".format(latest_model_path))

        ## RUN SWA
        temp_l_SetPath = temp_al_args[1]
        temp_u_SetPath = temp_al_args[3]

        print(
            "Best Val Acc: {}, Best Val Epoch: {}".format(
                best_val_acc, best_val_epoch + 1
            )
        )
        print("============================")
        best_val_accuracies.append(best_val_acc)
        best_val_epochs.append(best_val_epoch)

        if args.swa_mode and args.swa_freq > 0:
            # This means we want to run SWA else we won't

            args.device_ids = np.arange(cfg.NUM_GPUS)

            swa_temp_out_dir = temp_out_dir  # + "checkpoints/"
            if swa_temp_out_dir.find("checkpoints") != -1:
                # remove checkpoints directory
                swa_temp_out_dir = swa_temp_out_dir[
                    : swa_temp_out_dir.index("checkpoints")
                ]
            swa_temp_out_dir = os.path.join(swa_temp_out_dir, "checkpoints/")
            argListSWA = [
                args,
                latest_model_path,
                temp_l_SetPath,
                temp_u_SetPath,
                swa_temp_out_dir,
                trainDataset,
                noAugDataset,
                cfg,
            ]

            print("RUNNING SWA FROM HERE ..........")
            print("Check data paths")
            print("LsetPath: ", cfg.ACTIVE_LEARNING.LSET_PATH)
            print("UsetPath: ", cfg.ACTIVE_LEARNING.USET_PATH)

            SWA_subprocess_call(argListSWA, debug=True)

        print("=====BEST MODEL=====")
        print("temp_out_dir: {}".format(temp_out_dir))
        temp_al_start = True if i == 0 else False
        if i == 0:
            best_model_path = get_best_model_path(
                args, data_splits, i, directory_specific, temp_al_start
            )
        else:
            best_model_path = get_best_model_path(
                args,
                data_splits,
                i,
                directory_specific,
                temp_al_start,
                directPath=temp_out_dir,
            )

        print("best_model_path: {}".format(best_model_path))

        if cfg.TRAIN.TRANSFER_EXP:
            import copy

            temp_cfg = copy.deepcopy(cfg)
            temp_cfg.OUT_DIR = cu.get_checkpoint_dir_wo_checkpoint(cfg)
            print("cfg.OUT_DIR : {}".format(cfg.OUT_DIR))
            print("temp_cfg.OUT_DIR: {}".format(temp_cfg.OUT_DIR))
            temp_cfg.freeze()
            custom_dump_cfg(temp_cfg=temp_cfg)
            temp_cfg.defrost()

        ##test model via subprocess
        temp_test_acc = test_net_subprocess_call(
            temp_out_dir, best_model_path, debug=True
        )
        test_accuracies.append(temp_test_acc)
        test_model_paths.append(best_model_path)

    if al_max_iter > 1:
        for i in range(len(data_splits) - 1):
            # print("For {}% split, best val accuracy: {} achieved at epoch: {}"\
            #    .format(data_splits[i+1], best_val_accuracies[i], best_val_epochs[i]))
            print(
                "For {}% split, test accuracy: {:.3f} where model was loaded from path: {}".format(
                    data_splits[i + 1], test_accuracies[i], test_model_paths[i]
                )
            )


if __name__ == "__main__":
    from pycls.core.config import cfg

    main(cfg)
