import torch
import pickle
import sys
import os
import pycls.utils.logging as lu

import pycls.core.model_builder as model_builder

from helper.subprocess_utils import vaal_sampling_util

import numpy as np
import copy


logger = lu.get_logger(__name__)


def save_plot_values(
    temp_arrays, temp_names, cfg, isParallel=True, saveInTextFormat=False
):

    """Saves arrays provided in the list in npy format"""
    # return if not master process
    if isParallel:
        if not du.is_master_proc():
            return

    for i in range(len(temp_arrays)):
        temp_arrays[i] = np.array(temp_arrays[i])
        if not os.path.exists(cfg.OUT_DIR):
            os.makedirs(cfg.OUT_DIR)
        if saveInTextFormat:
            print(
                f"Saving {temp_names[i]} at {cfg.OUT_DIR+temp_names[i]}.txt in text format!!"
            )
            np.savetxt(cfg.OUT_DIR + temp_names[i] + ".txt", temp_arrays[i], fmt="%d")
        else:
            print(
                f"Saving {temp_names[i]} at {cfg.OUT_DIR+temp_names[i]}.npy in numpy format!!"
            )
            np.save(cfg.OUT_DIR + temp_names[i] + ".npy", temp_arrays[i])

        # print("Saved {} successfully at {}!!".format(temp_names[i], cfg.OUT_DIR+temp_names[i]+".npy"))


def active_set_performance(
    cfg, dataObj, clf, noAugDataset, activeSet, model_path
) -> float:
    """Calculates performance on the active set sampled by any AL method.

    Accuracy is evaluated by considering the oracle labels as Ground truth and predictions
    as the model predictions on active set where the model is the one trained in previous AL iteration.
    For example, consider we perform active sampling from 20% to 30% data and lets say [X1...XM] is
    the active set sampled. Then we consider predictions as model (trained on 20% data) predictions.

    Intuitively we want to examine how much we actually gain by inferring the labels from oracle.

    NOTE: This is a part of an ablation experiment which was not reported in the main paper.

    Args:
        cfg: Reference to the config yaml.
        dataObj: Reference to the data class.
        clf: Image classifier.
        noAugDataset: reference to the dataset with no-augmentations applied.
        activeSet: Active set sampled.
        model_path: Path to the model checkpoint used in AL sampling.

    Returns:
        float: Accuracy on the active set.
    """

    tp = 0
    n = 0

    old_clfmode = clf.training
    clf.eval()

    activeSetLoader = dataObj.getSequentialDataLoader(
        indexes=activeSet, batch_size=int(cfg.TRAIN.BATCH_SIZE), data=noAugDataset
    )
    onlyOnce = True
    with torch.no_grad():
        for i, (as_data, as_label) in enumerate(activeSetLoader):

            if onlyOnce and cfg.TRAIN.DATASET == "CIFAR10":
                c10_classes = (
                    "plane",
                    "car",
                    "bird",
                    "cat",
                    "deer",
                    "dog",
                    "frog",
                    "horse",
                    "ship",
                    "truck",
                )
                import matplotlib.pyplot as plt

                plt.title(f"Class: {c10_classes[as_label[0]]}")
                # plt.plot(as_data[0])
                plt.imshow(np.transpose(as_data[0].numpy(), (1, 2, 0)))
                plt.savefig(os.path.join(cfg.OUT_DIR, "temp_photo.png"))
                onlyOnce = False
            as_data = as_data.cuda(0)
            as_label = as_label.cuda(0)

            output = clf(as_data)
            _, prediction = torch.max(output, dim=1)

            tp += torch.sum(prediction == as_label).item()
            n += as_label.shape[0]

    activeSetAccuracy = 100.0 * float(tp) / float(n)
    print(
        f"For {cfg.ACTIVE_LEARNING.SAMPLING_FN} sampling, activeSet accuracy: ",
        activeSetAccuracy,
    )

    activeSetPerformance_fpath = os.path.join(cfg.OUT_DIR, "activeSet_performance.txt")
    fp = open(activeSetPerformance_fpath, "w")

    fp.write(
        f"Accuracy of the model on activeSet: {activeSetAccuracy} % out of 100%.\n"
    )
    fp.write(f"Model loaded from path: {model_path}")
    fp.write(f"\n[Activeset stats] Correct: {tp} and total: {n} .")

    fp.close()

    clf.train(old_clfmode)

    return activeSetAccuracy


def active_sampling(cfg) -> None:
    """Implements Active sampling.

    Args:
        cfg: Reference to config yaml.

    Returns: None
    """

    # with torch.no_grad():
    model_path = cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR
    model = model_builder.build_model(
        cfg, active_sampling=cfg.ACTIVE_LEARNING.ACTIVATE, isDistributed=False
    )

    checkpoint = torch.load(model_path, map_location="cpu")

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        isModuleStrPresent = False
        # remove module
        for k in checkpoint.keys():
            if k.find("module.") == -1:
                continue
            isModuleStrPresent = True
            break

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

        model.load_state_dict(checkpoint)
    model.cuda(torch.cuda.current_device())
    logger.info("Loaded initial weights from: {}".format(model_path))

    # load dataset
    # We can load noAug dataset as we just need datapoints (not their labels)
    # for uSet and lSet
    # However if you make use of labels in active sampling then do not load dataset references
    # rather pass it from main otherwise for noisy experiments results will not be consistent
    from al_utils.data import Data as custom_Data

    if cfg.TRAIN.DATASET in ["CIFAR10", "CIFAR100", "SVHN", "MNIST", "STL10"]:
        # No need to pass randaug true here as for AL we never intent to use data augmentations
        dataObj = custom_Data(dataset=cfg.TRAIN.DATASET)

        dataObj.eval_mode = True  # To load the data w/o any aug
        noAugDataset, _ = dataObj.getDataset(
            save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True
        )

        lSet, uSet, valSet = dataObj.loadPartitions(
            lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH,
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH,
            valSetPath=cfg.ACTIVE_LEARNING.VALSET_PATH,
        )
    else:
        # For imagenet
        # noAugDataset = None
        # dataObj = None
        dataObj = custom_Data(dataset=cfg.TRAIN.DATASET)

        dataObj.eval_mode = True  # To load the data w/o any aug
        noAugDataset, _ = dataObj.getDataset(
            save_dir=cfg.TRAIN_DIR, isTrain=True, isDownload=True
        )

        lSet = np.load(cfg.ACTIVE_LEARNING.LSET_PATH, allow_pickle=True)
        uSet = np.load(cfg.ACTIVE_LEARNING.USET_PATH, allow_pickle=True)
        valSet = np.load(cfg.ACTIVE_LEARNING.VALSET_PATH, allow_pickle=True)

    from al_utils.ActiveLearning import ActiveLearning

    print(
        "Perform {} sampling through subprocess".format(cfg.ACTIVE_LEARNING.SAMPLING_FN)
    )

    if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["vaal", "vaal_minus_disc"]:
        return vaal_sampling_util(cfg, dataObj, debug=True)

    else:
        activelearning = ActiveLearning(dataObj=dataObj, cfg=cfg)

    with torch.no_grad():
        activeSet, uSet = activelearning.sample_from_uSet(
            clf_model=model, lSet=lSet, uSet=uSet, trainDataset=noAugDataset
        )
    print("========BEFORE==========")
    print("len(uSEt): ", len(uSet))
    print("len(lSEt): ", len(lSet))

    print("==================")
    lSet = np.append(lSet, activeSet)
    assert (
        len(set(valSet) & set(uSet)) == 0
    ), "Intersection is not allowed between validationset and uset"
    assert (
        len(set(valSet) & set(lSet)) == 0
    ), "Intersection is not allowed between validationset and lSet"
    assert (
        len(set(uSet) & set(lSet)) == 0
    ), "Intersection is not allowed between uSet and lSet"
    print(
        "After including activeSet -- len(lSet): {} and len(uSet): {}".format(
            len(lSet), len(uSet)
        )
    )

    # Save numpy arrays
    print("saving pickle values...")
    save_plot_values(
        [lSet, uSet, activeSet], ["lSet", "uSet", "activeSet"], cfg, isParallel=False
    )
    print("saved!!")
    # Save index sets in txt formats
    print("saving text values...")
    save_plot_values(
        [lSet, uSet, activeSet],
        ["lSet", "uSet", "activeSet"],
        cfg,
        isParallel=False,
        saveInTextFormat=True,
    )
    print("saved!!")
    print("======AFTER AL============")
    print("ActiveSet: ", len(activeSet))
    print("len(uSet): ", len(uSet))
    print("len(lSet): ", len(lSet))

    if cfg.TRAIN.DATASET in ["CIFAR10", "CIFAR100", "SVHN", "MNIST", "STL10"]:
        active_set_performance(cfg, dataObj, model, noAugDataset, activeSet, model_path)


def ensemble_active_learning(cfg, ensemble_args):
    """Implements the ensemble based AL methods.

    Args:
        cfg: Reference to config file.
        ensemble_args: Args relevant for ensemble methods.
    """

    model_paths, num_ensembles, noAugDataset, dataObj, temp_out_dir = ensemble_args

    lSet = np.load(cfg.ACTIVE_LEARNING.LSET_PATH, allow_pickle=True)
    uSet = np.load(cfg.ACTIVE_LEARNING.USET_PATH, allow_pickle=True)
    valSet = np.load(cfg.ACTIVE_LEARNING.VALSET_PATH, allow_pickle=True)

    from al_utils.ActiveLearning import ActiveLearning

    print(
        "Perform {} sampling through subprocess".format(cfg.ACTIVE_LEARNING.SAMPLING_FN)
    )
    activelearning = ActiveLearning(dataObj=dataObj, cfg=cfg)

    all_models = []

    current_device = torch.cuda.current_device()

    for i in range(num_ensembles):
        print("=====================================")
        print(f"Trying to load model from {model_paths[i]}")
        import pycls.core.model_builder as model_builder

        temp_model = model_builder.build_model(
            cfg, active_sampling=cfg.ACTIVE_LEARNING.ACTIVATE, isDistributed=False
        )
        temp_ckpt = torch.load(model_paths[i], map_location="cpu")
        temp_model.load_state_dict(temp_ckpt["model_state"])
        temp_model.cuda(current_device)
        print("Loaded initial weights from: {}".format(model_paths[i]))
        print("=====================================")
        all_models.append(temp_model)

    with torch.no_grad():
        activeSet, uSet = activelearning.sample_from_uSet(
            clf_model=None,
            lSet=lSet,
            uSet=uSet,
            trainDataset=noAugDataset,
            supportingModels=all_models,
        )

    cfg.OUT_DIR = temp_out_dir  # Resetting original directory

    lSet = np.append(lSet, activeSet)
    assert (
        len(set(valSet) & set(uSet)) == 0
    ), "Intersection is not allowed between validationset and uset"
    assert (
        len(set(valSet) & set(lSet)) == 0
    ), "Intersection is not allowed between validationset and lSet"
    assert (
        len(set(uSet) & set(lSet)) == 0
    ), "Intersection is not allowed between uSet and lSet"
    print(
        "After including activeSet -- len(lSet): {} and len(uSet): {}".format(
            len(lSet), len(uSet)
        )
    )
    # Save numpy arrays

    save_plot_values(
        [lSet, uSet, activeSet], ["lSet", "uSet", "activeSet"], cfg, isParallel=False
    )
    # Save index sets in txt formats
    save_plot_values(
        [lSet, uSet, activeSet],
        ["lSet", "uSet", "activeSet"],
        cfg,
        isParallel=False,
        saveInTextFormat=True,
    )
    # sys.exit(0)


tempArgsFile = sys.argv[1]

# Getting back the objects:
with open(tempArgsFile, "rb") as f:  # Python 3: open(..., 'rb')
    cfg, ensemble_args = pickle.load(f)

if len(ensemble_args):
    print("=== Performing Ensemble based Active Learning ===")
    ensemble_active_learning(cfg, ensemble_args)
else:
    active_sampling(cfg)
