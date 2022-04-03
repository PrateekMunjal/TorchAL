import torch
import torchcontrib

import pycls.datasets.loader as imagenet_loader

from pycls.core.model_builder import build_model

from al_utils.data import Data as custom_Data
from tqdm import tqdm
import copy


def load_model(args, cfg, model_path, active_sampling=False, isDistributed=False):
    """Loads both the optimizer and model checkpoint from input model_path.

    Args:
        args: Reference to all input args.
        cfg: Reference to config yaml.
        model_path: Path to checkpoint.
        active_sampling (bool, optional): Switch to detect if we are loading model during AL sampling. Defaults to False.
        isDistributed (bool, optional): Switch to turn on distributed training. Defaults to False.

    Returns:
        torch.optim, torch.nn.Model: Return both the optimizer and model.
    """

    # Construct the model
    if args.model_type != cfg.MODEL.TYPE:
        cfg.MODEL.TYPE = args.model_type
        cfg.MODEL.DEPTH = args.model_depth
    model = build_model(cfg, active_sampling, isDistributed)

    model_type = args.model_type

    if cfg.OPTIM.TYPE == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wt_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wt_decay
        )

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

    # print(f"[After loading checkpoint] optimizer in Swa: {optimizer}")

    print("Curent_id: {}".format(current_id))
    model = model.cuda(current_id)

    model = torch.nn.DataParallel(
        model, device_ids=[i for i in range(torch.cuda.device_count())]
    )
    return optimizer, model


def get_validation_accuracy(valSetLoader, model, current_id):
    """Evaluates accuracy on validation set.

    Args:
        valSetLoader (torch.utils.data.DataLoader): dataloader itertating over validation set.
        model (torch.nn.Module): Image classifier.
        current_id (int): gpu_id.

    Returns:
        float: Accuracy on validation set.
    """
    totalDataPoints = 0.0
    totalCorrectClfPoints = 0.0
    valSetpbar = tqdm(valSetLoader, desc="Validation Accuracy")
    print("Number of val batches: ", len(valSetLoader))

    for i, (x, y) in enumerate(valSetpbar):
        x = x.cuda(current_id)
        y = y.cuda(current_id)

        pred = model(x)

        _, prediction = torch.max(pred.data, dim=1)
        totalDataPoints += y.size(0)
        totalCorrectClfPoints += (prediction == y).sum().item()

    accuracy = totalCorrectClfPoints / totalDataPoints
    accuracy *= 100.0
    return accuracy


def swa_train(
    args,
    load_model_path,
    lSetPath,
    uSetPath,
    save_path,
    trainDataset,
    noAugDataset,
    cfg,
):
    """Implements functionality of running SWA.

    Args:
        args: Input args.
        load_model_path (str): Path to the model checkpoint.
        lSetPath (str): Path to labeled set.
        uSetPath (str): Path to unlabeled set
        save_path (str): Path to save all the SWA results.
        trainDataset (torch.utils.data.Dataset): Reference to the training dataset.
        noAugDataset (torch.utils.data.Dataset): Reference to the no-augmentation dataset.
        cfg : Reference to config.
    """

    valSetPath = args.valSetPath
    save_epoch = args.clf_epochs + 1

    loss = torch.nn.CrossEntropyLoss()
    print("SWA_TRAIN on selected device_ids: {}".format(cfg.NUM_GPUS))

    current_id = torch.cuda.current_device()  # cfg.NUM_GPUS[0]
    print("~~~ current_id: {}".format(current_id))

    print("===== In swa_util =====")
    if args.dataset.upper() != cfg.TRAIN.DATASET:
        cfg.TRAIN.DATASET = args.dataset.upper()
    if not cfg.TRAIN.DATASET == "IMAGENET":
        print("===== trainDataset =====")
        print(trainDataset)

    if cfg.TRAIN.DATASET == "IMAGENET":
        print("=======================================")
        print("===== Loading Imagenet Dataloaders ====")
        print("[prev] num workers: ", cfg.DATA_LOADER.NUM_WORKERS)
        cfg.DATA_LOADER.NUM_WORKERS = 0
        print("[now] num workers: ", cfg.DATA_LOADER.NUM_WORKERS)
        print("=======================================")

        lSetLoader, valSetLoader = imagenet_loader.get_data_loaders(
            cfg, isDistributed=False
        )

    else:
        dataObj = custom_Data(dataset=args.dataset)  # , israndAug=args.rand_aug)

        lSet, _, valSet = dataObj.loadPartitions(
            lSetPath=lSetPath, uSetPath=uSetPath, valSetPath=valSetPath
        )

        lSetLoader = dataObj.getIndexesDataLoader(
            indexes=lSet, batch_size=int(args.train_batch_size), data=trainDataset
        )

        bn_lSetLoader = dataObj.getIndexesDataLoader(
            indexes=lSet, batch_size=int(args.train_batch_size), data=noAugDataset
        )

        oldmode = dataObj.eval_mode
        dataObj.eval_mode = True
        print("-------JUST BEFORE LOADING VALSET LOADER--------")
        print(f"dataObj.eval_mode: {dataObj.eval_mode}")
        valSetLoader = dataObj.getIndexesDataLoader(
            indexes=valSet, batch_size=int(args.train_batch_size), data=noAugDataset
        )

        dataObj.eval_mode = oldmode

        print("-------VALSET LOADER---------")
        print(valSetLoader)

    optimizer, model = load_model(
        args, cfg, load_model_path, active_sampling=False, isDistributed=False
    )

    print(f"After Loading model. optimizer: {optimizer}")
    model.eval()

    print(
        "Evaluating model validaton accuracy to confirm whether model is correctly loaded"
    )
    temp_acc = get_validation_accuracy(valSetLoader, model, current_id)
    print("Newly loaded model has accuracy: {}".format(temp_acc))

    model.train()

    swa_optimizer = torchcontrib.optim.SWA(
        optimizer, swa_start=args.swa_iter, swa_freq=args.swa_freq, swa_lr=args.swa_lr
    )

    print(f"SWA Optimizer: {swa_optimizer}")

    print("Training SWA for {} epochs.".format(args.swa_epochs))
    temp_max_itrs = len(lSetLoader)
    print(f"len(lSetLoader): {len(lSetLoader)}")
    temp_cur_itr = 0
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
            # if(temp_cur_itr % 50 == 0):
            print(f"Iteration [{temp_cur_itr}/{temp_max_itrs}] Done !!")

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

    # Record the state
    checkpoint = {
        "epoch": save_epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": None,
    }
    # Write the checkpoint

    checkpoint_file = save_path + "[SWA]valSet_acc_{}_model_epoch_{:04}.pyth".format(
        accuracy, save_epoch
    )
    print("---Before SAVING SWA MODEL----")
    torch.save(checkpoint, checkpoint_file)
    print("SAVED SWA model")
    print("SWA Model saved at {}".format(checkpoint_file))
    return
