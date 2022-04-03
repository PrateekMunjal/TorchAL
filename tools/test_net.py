"""Test a trained classification model."""

import argparse
import numpy as np
import os
import sys
import torch

from pycls.core.config import assert_cfg

# from pycls.core.config import cfg
from pycls.utils.meters import TestMeter

import pycls.datasets.loader as imagenet_loader

import pycls.core.model_builder as model_builder
import pycls.datasets.loader as loader
import pycls.utils.checkpoint as cu
import pycls.utils.distributed as du
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.multiprocessing as mpu

from al_utils.data import Data as custom_Data

logger = lu.get_logger(__name__)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Test a trained classification model")
    parser.add_argument("--cfg", dest="cfg_file", help="Config file", type=str)
    parser.add_argument(
        "opts",
        help="See pycls/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--model_path_file",
        type=str,
        default="",
        help="Path of file containing model paths",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def log_model_info(model):
    """Logs model info"""
    # print('Model:\n{}'.format(model))
    print("Params: {:,}".format(mu.params_count(model)))
    print("Flops: {:,}".format(mu.flops_count(model)))


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch, cfg):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.0
    totalSamples = 0.0
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.scaled_all_reduce(cfg, [top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()

        # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
        misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
        totalSamples += inputs.size(0) * cfg.NUM_GPUS

        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(top1_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()

    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    return misclassifications / totalSamples


def test_model(test_acc, cfg):
    """Evaluates the model."""

    # Build the model (before the loaders to speed up debugging)
    model = model_builder.build_model(
        cfg, active_sampling=cfg.ACTIVE_LEARNING.ACTIVATE, isDistributed=True
    )
    log_model_info(model)

    # Load model weights
    cu.load_checkpoint(cfg, cfg.TEST.WEIGHTS, model)
    print("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))

    # Create data loaders
    # test_loader = loader.construct_test_loader()
    if cfg.TRAIN.DATASET == "IMAGENET":
        test_loader = imagenet_loader.construct_test_loader(cfg)
    else:
        dataObj = custom_Data(dataset=cfg.TRAIN.DATASET)

        # print("=========== Loading testDataset ============")
        was_eval = dataObj.eval_mode
        dataObj.eval_mode = True
        testDataset, _ = dataObj.getDataset(
            save_dir=cfg.TEST_DIR, isTrain=False, isDownload=True
        )
        dataObj.eval_mode = was_eval

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

    # Create meters
    test_meter = TestMeter(len(test_loader), cfg)

    # Evaluate the model
    test_err = test_epoch(test_loader, model, test_meter, 0, cfg)
    print("Test Accuracy: {:.3f}".format(100.0 - test_err))

    if cfg.NUM_GPUS > 1:
        test_acc.value = 100.0 - test_err
    else:
        return 100.0 - test_err


def test_single_proc_test(test_acc, cfg):
    """Performs single process evaluation."""

    # Setup logging
    lu.setup_logging(cfg)
    # Show the config
    # print('Config:\n{}'.format(cfg))

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Evaluate the model
    if cfg.NUM_GPUS > 1:
        test_model(test_acc, cfg)
    else:
        return test_model(test_acc, cfg)


def test_main(args, avail_nGPUS=4):
    from pycls.core.config import cfg

    test_acc = 0.0
    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    # cfg.PORT = 10095

    assert_cfg()
    # avail_nGPUS = torch.cuda.device_count()
    if cfg.NUM_GPUS > avail_nGPUS:
        print(
            "Available GPUS at test machine: ",
            avail_nGPUS,
            " but requested config has GPUS: ",
            cfg.NUM_GPUS,
        )
        print(f"Running on {avail_nGPUS} instead of {cfg.NUM_GPUS}")
        cfg.NUM_GPUS = avail_nGPUS

    cfg.freeze()

    dataset = cfg.TEST.DATASET
    data_split = cfg.ACTIVE_LEARNING.DATA_SPLIT
    seed_id = cfg.RNG_SEED
    sampling_fn = cfg.ACTIVE_LEARNING.SAMPLING_FN

    print("======================================")
    print("~~~~~~ CFG.NUM_GPUS: ", cfg.NUM_GPUS)
    print("======================================")

    # Perform evaluation
    if cfg.NUM_GPUS > 1:
        test_acc = mpu.multi_proc_run_test(
            num_proc=cfg.NUM_GPUS, fun=test_single_proc_test, fun_args=(cfg,)
        )
    else:
        temp_acc = 0.0
        test_acc = test_single_proc_test(temp_acc, cfg)

    # Save test accuracy
    test_model_path = cfg.TEST.WEIGHTS
    test_model_name = np.array([test_model_path.split("/")[-1]])
    file_name = "test_acc_"
    file_save_path = cfg.OUT_DIR
    if cfg.TRAIN.TRANSFER_EXP:
        file_save_path = os.path.abspath(os.path.join(file_save_path, os.pardir))
        # file_save_path= os.path.join(file_save_path,os.path.join("transfer_experiment",cfg.MODEL.TRANSFER_MODEL_TYPE+"_depth_"+str(cfg.MODEL.TRANSFER_MODEL_DEPTH)))#+"/"
    file_save_path = os.path.join(file_save_path, file_name)
    test_accuracy = np.array([test_acc], dtype="float")

    temp_data = np.column_stack((test_model_path, test_accuracy))
    # print(temp_data.shape)

    np.savetxt(file_save_path + ".txt", temp_data, delimiter=" ", fmt="%s")
    np.save(file_save_path + ".npy", temp_data)
    print(f"Test accuracy [npy|txt] are saved at {file_save_path}")

    return test_acc, dataset, data_split, seed_id, sampling_fn


def get_config_model_paths(model_path):
    config_path_dir = model_path.split("checkpoints")[0]
    model_detail = os.path.split(model_path)[1]
    config_path_file = os.path.join(config_path_dir, "config.yaml")
    return config_path_file, model_detail


def main():
    # Parse cmd line args
    args = parse_args()

    # Retain old functionality
    if args.model_path_file == "":
        import subprocess as sp

        os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
        temp_pythonPath = os.popen("which python").readlines()[0].rstrip()
        avail_nGPUS = sp.check_output(
            (temp_pythonPath, "-c", "import torch; print(torch.cuda.device_count())")
        )
        print("avail_nGPUS: ", avail_nGPUS)

        avail_nGPUS = int(avail_nGPUS)
        temp_test_acc, dataset, data_split, seed_id, sampling_fn = test_main(
            args, avail_nGPUS
        )  # =4)
        print("==In main function==")
        print(
            f"[Acquisition:{sampling_fn};Seed:{seed_id}]Test accuracy on {dataset} using {data_split}% of data is {temp_test_acc}"
        )
        return

    fname = args.model_path_file
    file_path = os.path.join(os.getcwd(), fname)
    print("dir_fname: {}".format(file_path))

    lc = 0
    temp_model_path = ""
    test_accuracies = []
    dataset_names = []
    data_splits = []
    seed_ids = []
    sampling_methods = []
    # print('file_path: ', file_path)
    file_path = file_path.encode("utf-8").strip()
    with open(file_path) as f:
        tempLine = f.readline()
        lc += 1
        while tempLine:
            if lc == 1:
                # starts with baseclf
                assert tempLine.startswith(
                    "baseclf"
                ), "The first line should start with {}: string".format("baseclf")
                temp_model_path = tempLine.split("baseclf:")[1].rsplit("\n")[0]
            else:
                temp_model_path = tempLine.rsplit("\n")[0]
            # print(temp_model_path)
            cfg_path, model_detail = get_config_model_paths(temp_model_path)
            print("cfg_path: {}".format(cfg_path))
            print("model_path: {}".format(temp_model_path))
            args.cfg_file = cfg_path
            # hardcoded port so that while running multiple experiments the testing subprocess never fails.
            args.opts = ["TEST.WEIGHTS", temp_model_path, "PORT", 10787]

            logger.info("Before running test_main function")
            temp_test_acc, dataset, data_split, seed_id, sampling_fn = test_main(args)

            # model_names.append(model_detail)
            test_accuracies.append(temp_test_acc)
            dataset_names.append(dataset)
            data_splits.append(data_split)
            seed_ids.append(seed_id)
            sampling_methods.append(sampling_fn)

            tempLine = f.readline()
            lc += 1

    print("Baseclf: test accuracy {}".format(test_accuracies[0]))

    for i in range(1, len(test_accuracies)):
        print(
            "On {} with {}% data and seed: {}, {} method obtains test accuracy: {}".format(
                dataset_names[i],
                data_splits[i],
                seed_ids[i],
                sampling_methods[i],
                test_accuracies[i],
            )
        )


if __name__ == "__main__":
    main()
