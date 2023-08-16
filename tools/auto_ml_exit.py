import torch
import torchvision
import pickle
import sys
import os
import pycls.utils.logging as lu

import pycls.core.model_builder as model_builder

import numpy as np
import copy

from al_utils.data import Data as custom_Data

from helper.path_extractor import search_best_model_path

import optuna

from train_al import single_proc_train

from pycls.core.config import custom_assert_cfg, custom_dump_cfg

import pycls.utils.multiprocessing as mpu
from glob import glob

import plotly.io as pio

pio.orca.config.use_xvfb = True

import time

logger = lu.get_logger(__name__)


def al_main(cfg, args, trainDataset, valDataset, dataObj, trial, isPruning):
    """Main function to facilitate AL settings."""

    custom_assert_cfg(cfg)

    # Ensure that the output dir exists
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    # Save the config
    custom_dump_cfg(cfg)
    print("config", cfg)
    if cfg.NUM_GPUS > 1:
        print("============================")
        print("Number of Gpus available for multiprocessing: {}".format(cfg.NUM_GPUS))
        print("============================")
        best_val_acc, best_val_epoch = mpu.multi_proc_run(
            num_proc=cfg.NUM_GPUS,
            fun=single_proc_train,
            fun_args=(trainDataset, valDataset, dataObj, cfg, trial, isPruning),
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
            trial,
            isPruning,
        )

    return best_val_acc, best_val_epoch


def objective(trial, cfg, args, out_dir_path, isPruning):
    """Runs an AutoML trial and returns the best performance on validation set.

    Args:
        trial: the trial number
        cfg: Reference to config yaml
        args: Reference to input args
        out_dir_path (str): Path to save results for current trial.
        isPruning (bool): Switch to prune the trial.

    Raises:
        NotImplementedError: when dataset is not supported.
        NotImplementedError: when dataset is IMAGENET because we do not support AutoML on Imagenet.

    Returns:
        float: Best performance on the validation set.
    """

    cfg.OUT_DIR = os.path.join(out_dir_path, f"trial-{trial.number}") + "/"

    if os.path.exists(os.path.join(cfg.OUT_DIR, "checkpoints")):
        checkpoint_path = os.path.join(cfg.OUT_DIR, "checkpoints")
        return search_best_model_path(temp_path=checkpoint_path, isDirectPath=True)

    ## Assertions
    assert cfg.TRAIN.DATASET.lower() == args.dataset.lower()
    assert cfg.RANDAUG.ACTIVATE == args.rand_aug
    assert (
        cfg.MODEL.NUM_CLASSES == args.num_classes
    ), f"Found cfg.MODEL.NUM_CLASSES: {cfg.MODEL.NUM_CLASSES} and args.num_classes: {args.num_classes} different, they should be same."
    assert cfg.TRAIN.EVAL_PERIOD == args.eval_period

    al_model_phase = cfg.ACTIVE_LEARNING.ACTIVATE
    print("== al_model_phase: {} ==".format(al_model_phase))
    al_start = args.init_partition

    if al_model_phase == False:
        sampling_fn = None
        data_splits = [args.init_partition]
    else:
        assert cfg.ACTIVE_LEARNING.SAMPLING_FN == args.sampling_fn
        sampling_fn = args.sampling_fn

        # construct data splits
        al_step = args.step_partition
        al_stop = al_start + args.al_max_iter * al_step
        data_splits = [round(i, 1) for i in np.arange(al_start, al_stop, al_step)]

    al_max_iter = len(data_splits)

    if args.reuse_aml:
        print("-------- Reusing automl -------")
        print("cfg.OUT_DIR: ", cfg.OUT_DIR)
        print("cfg.OUT_DIR.split: ", cfg.OUT_DIR.split("auto_ml_results/"))
        prefix_out_dir, suffix_out_dir = cfg.OUT_DIR.split("auto_ml_results/")
        temp_load_cfg_DIR = os.path.join(
            prefix_out_dir, "best_automl_results", suffix_out_dir
        )
        print("temp_load_cfg_DIR: ", temp_load_cfg_DIR)
        prefix_out_dir, suffix_out_dir = temp_load_cfg_DIR.split("start_")
        load_cfg_fpath = os.path.join(
            prefix_out_dir, f"start_{args.reuse_aml_seed}", suffix_out_dir[2:]
        )
        load_cfg_fpath = load_cfg_fpath.split("trial-")[0]
        load_cfg_fpath = os.path.join(load_cfg_fpath, "config.yaml")
        print("load_cfg_fpath: ", load_cfg_fpath)
        temp_fp = open(load_cfg_fpath)
        import yaml

        load_cfg_data_temp = yaml.load_all(temp_fp, Loader=yaml.FullLoader)
        for aml_cfg_data in load_cfg_data_temp:
            break

        lr = aml_cfg_data["OPTIM"]["BASE_LR"]
        wd = aml_cfg_data["OPTIM"]["WEIGHT_DECAY"]
        batch_size = aml_cfg_data["TRAIN"]["BATCH_SIZE"]
        optimizer = aml_cfg_data["OPTIM"]["TYPE"]
        if cfg.RANDAUG.ACTIVATE:
            RA_N = aml_cfg_data["RANDAUG"]["N"]
            RA_M = aml_cfg_data["RANDAUG"]["M"]

    else:
        # auto-ml hyperparams
        lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        wd = trial.suggest_loguniform("weight_decay", 1e-8, 1e-3)
        batch_size = trial.suggest_categorical(
            "batch_size", [2**i for i in range(3, 10)]
        )
        # batch_size = trial.suggest_categorical('batch_size', [512,1024,256])
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "ADAM"])
        if cfg.RANDAUG.ACTIVATE:
            RA_N = trial.suggest_int("RA_N", 1, 15)
            RA_M = trial.suggest_int("RA_M", 1, 9)

    print(f"======== Hyper-params for TRIAL: {trial.number} ========")
    print(f"Learning Rate: {lr}")
    print(f"Weight Decay : {wd}")
    print(f"Batch Size   : {batch_size}")
    print(f"Optimizer    : {optimizer}")
    if cfg.RANDAUG.ACTIVATE:
        print(f"RA_N         : {RA_N}")
        print(f"RA_M         : {RA_M}")
    print("==================================================")

    ##
    if cfg.RANDAUG.ACTIVATE:
        args.rand_aug_N = RA_N
        args.rand_aug_M = RA_M

    # Make changes to config file
    cfg.OPTIM.BASE_LR = lr
    cfg.OPTIM.WEIGHT_DECAY = wd
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.OPTIM.TYPE = optimizer
    if cfg.RANDAUG.ACTIVATE:
        cfg.RANDAUG.N = RA_N
        cfg.RANDAUG.M = RA_M

    cfg.RNG_SEED = args.seed_id

    # load datasets
    if args.dataset in ["CIFAR10", "CIFAR100", "SVHN", "MNIST", "STL10", "RSNA"]:
        dataObj = custom_Data(
            dataset=cfg.TRAIN.DATASET, israndAug=cfg.RANDAUG.ACTIVATE, args=args
        )
        print("==== Loading trainDataset ====")
        trainDataset, n_TrainDatapts = dataObj.getDataset(
            save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True
        )

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
    else:
        print(f"{args.dataset} dataset not handled yet.")
        raise NotImplementedError

    # Because this happens after active learning process then
    if cfg.ACTIVE_LEARNING.ACTIVATE and cfg.ACTIVE_LEARNING.NOISY_ORACLE > 0.0:

        if cfg.TRAIN.DATASET == "IMAGENET":
            raise NotImplementedError

        print("============= ADDING NOISE =============")
        noise_percent = cfg.ACTIVE_LEARNING.NOISY_ORACLE
        # temp_data_split = cfg.ACTIVE_LEARNING.DATA_SPLIT
        # print('cfg.ACTIVE_LEARNING.LSET_PATH: ', cfg.ACTIVE_LEARNING.LSET_PATH)
        temp_activeset_fpath = os.path.abspath(
            os.path.join(cfg.ACTIVE_LEARNING.LSET_PATH, os.pardir)
        )
        temp_activeset_fpath = os.path.join(temp_activeset_fpath, "activeSet.npy")

        # print(f'accessing activeset at {os.path.join(cfg.OUT_DIR, "activeSet.npy")}')
        # activeSet = np.load(os.path.join(cfg.OUT_DIR, 'activeSet.npy'), allow_pickle=True)

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

    best_val_acc, best_val_epoch = al_main(
        cfg, args, trainDataset, valDataset, dataObj, trial, isPruning
    )  # , al_args)
    return best_val_acc


def run_auto_ml(cfg, args):
    """This function implements AutoML search.

    Args:
        cfg: Reference to config yaml
        args: Reference to input args
    """
    isPruning = True
    if isPruning:
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(),
            direction="maximize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
        )
    else:
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(), direction="maximize"
        )

    print("Sampler used: ", type(study.sampler).__name__)

    temp_out_dir = cfg.OUT_DIR

    # n_jobs: for cpu parallelization

    start_time = time.time()
    study.optimize(
        lambda trial: objective(trial, cfg, args, temp_out_dir, isPruning),
        n_trials=args.num_aml_trials,
        n_jobs=1,
    )

    end_time = time.time()
    print("=================")
    print(f"Time taken to finish study: {end_time - start_time} seconds")
    print("==================")

    # hparams_importances = optuna.importance.get_param_importances(study)

    complete_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE
    ]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # fig = optuna.visualization.plot_param_importances(study)
    # fig_savepath = os.path.join(cfg.OUT_DIR, f'{cfg.TRAIN.DATASET}-hparam-importance.png')
    # fig.write_image(fig_savepath)


def auto_ml_experiment_exists(cfg, args):
    """Boolean utility to check if automl results already exists.

    Args:
        cfg: Reference to config yaml file.
        args: Reference to input args.

    Returns:
        bool: Whether automl results exists or not.
    """

    check_dir = cfg.OUT_DIR
    num_trials = args.num_aml_trials

    print("~~ check_dir: ", check_dir)

    all_dirs = glob(os.path.join(check_dir, "trial-*"))
    print(f"==> Expected number of trials: {num_trials}")
    print(f"==> Found number of trials: {len(all_dirs)}")
    return len(all_dirs) == num_trials


tempArgsFile = sys.argv[1]

# Getting back the objects:
with open(tempArgsFile, "rb") as f:  # Python 3: open(..., 'rb')
    cfg, args = pickle.load(f)

# check if auto-ml experiments are already there
if not auto_ml_experiment_exists(cfg, args):
    run_auto_ml(cfg, args)
