import torch
from pycls.models import vaal_model as vm
import os
import numpy as np

import pycls.datasets.loader as imagenet_loader

from tqdm import tqdm

bce_loss = torch.nn.BCELoss().cuda()


def data_parallel_wrapper(model, cur_device, cfg):
    """Wraps a given model into pytorch data parallel.

    Args:
        model (torch.nn.Module): Image classifier.
        cur_device (int): gpu_id
        cfg : Reference to config.

    Returns:
        torch.nn.Dataparallel: model wrapped in dp.
    """
    model.cuda(cur_device)
    # model = torch.nn.DataParallel(model, device_ids = [cur_device])
    # assert cfg.NUM_GPUS == torch.cuda.device_count(), f"Expected torch device count (i.e {torch.cuda.device_count()}) same as number of gpus (i.e {cfg.NUM_GPUS}) in config file"
    model = torch.nn.DataParallel(
        model, device_ids=[i for i in range(torch.cuda.device_count())]
    )
    return model


def distributed_wrapper(cfg, model, cur_device):
    """Wraps a given model into pytorch distributed data parallel.

    Args:
        cfg : Reference to config.
        model (torch.nn.Module): Image classifier.
        cur_device (int): gpu_id

    Returns:
        torch.nn.parallel.DistributedDataParallel: model wrapped in ddp.
    """

    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)

    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model


def read_data(dataloader, labels=True):
    """Load infinite batches from dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader.
        labels (bool, optional): Switch to return either supervised batch or unsupervised batch. Defaults to True.

    """
    if labels:
        while True:
            for img, label in dataloader:
                yield img, label
    else:
        while True:
            for img, _ in dataloader:
                yield img


def vae_loss(x, recon, mu, logvar, beta):
    """Standard VAE Loss.

    Args:
        x: input image.
        recon: model output as reconstruction of input.
        mu: mean of latent.
        logvar: logvar of latent distribution.
        beta : Hyper-param in loss function.

    Returns:
        float: Loss over minibatches.
    """
    mse_loss = torch.nn.MSELoss().cuda()
    recon = recon.cuda()
    x = x.cuda()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD


def train_vae_disc_epoch(
    cfg,
    vae_model,
    disc_model,
    optim_vae,
    optim_disc,
    lSetLoader,
    uSetLoader,
    cur_epoch,
    n_lu,
    curr_vae_disc_iter,
    max_vae_disc_iters,
    change_lr_iter,
    isDistributed=False,
):
    """Trains VAE and Discriminator model for an epoch."""
    if isDistributed:
        lSetLoader.sampler.set_epoch(cur_epoch)
        uSetLoader.sampler.set_epoch(cur_epoch)

    print("len(lSetLoader): {}".format(len(lSetLoader)))
    print("len(uSetLoader): {}".format(len(uSetLoader)))

    labeled_data = read_data(lSetLoader)
    unlabeled_data = read_data(uSetLoader, labels=False)

    vae_model.train()
    disc_model.train()

    temp_bs = int(cfg.VAAL.VAE_BS)
    # train_iterations = cfg.VAAL.VAE_EPOCHS * int(n_lu/temp_bs)
    train_iterations = int(n_lu / temp_bs)

    print(f"cfg.VAAL.VAE_LR: {cfg.VAAL.VAE_LR}")
    print(f"cfg.VAAL.DISC_LR: {cfg.VAAL.DISC_LR}")
    print(f"cfg.VAAL.BETA: {cfg.VAAL.BETA}")
    print(f"cfg.VAAL.Z_DIM: {cfg.VAAL.Z_DIM}")
    print(f"cfg.VAAL.ADVERSARY_PARAM: {cfg.VAAL.ADVERSARY_PARAM}")
    for temp_iter in range(train_iterations):
        # KEEP THE BELOW LINES COMMENTED - WE found this trick used in vaal codebase.
        # Also vaal authors have now removed it but it still exists in commit history in the initial commit.
        # if curr_vae_disc_iter is not 0 and curr_vae_disc_iter % change_lr_iter == 0:
        #     #print("Changing LR ---- ))__((---- ")
        #     for param in optim_vae.param_groups:
        #         param['lr'] = param['lr'] * 0.9

        #     for param in optim_disc.param_groups:
        #         param['lr'] = param['lr'] * 0.9

        curr_vae_disc_iter += 1

        # if temp_iter is not 0 and temp_iter%30==0:
        #     break
        ## VAE Step
        disc_model.eval()
        vae_model.train()
        # print("temp_iter: {}".format(temp_iter))
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)

        labeled_imgs = labeled_imgs.type(torch.cuda.FloatTensor)
        unlabeled_imgs = unlabeled_imgs.type(torch.cuda.FloatTensor)

        labeled_imgs = labeled_imgs.cuda()
        unlabeled_imgs = unlabeled_imgs.cuda()

        # print(f'Iteration [{temp_iter}]: Lab imgs: {labeled_imgs.shape} \t Unlab imgs: {unlabeled_imgs.shape}')

        recon, z, mu, logvar = vae_model(labeled_imgs)

        unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, cfg.VAAL.BETA)
        unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae_model(unlabeled_imgs)
        transductive_loss = vae_loss(
            unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, cfg.VAAL.BETA
        )

        labeled_preds = disc_model(mu)
        unlabeled_preds = disc_model(unlab_mu)

        lab_real_preds = torch.ones(labeled_imgs.size(0), 1).cuda()
        unlab_real_preds = torch.ones(unlabeled_imgs.size(0), 1).cuda()
        dsc_loss = bce_loss(labeled_preds, lab_real_preds) + bce_loss(
            unlabeled_preds, unlab_real_preds
        )

        total_vae_loss = (
            unsup_loss + transductive_loss + cfg.VAAL.ADVERSARY_PARAM * dsc_loss
        )

        optim_vae.zero_grad()
        total_vae_loss.backward()
        optim_vae.step()

        ##DISC STEP
        vae_model.eval()
        disc_model.train()

        with torch.no_grad():
            _, _, mu, _ = vae_model(labeled_imgs)
            _, _, unlab_mu, _ = vae_model(unlabeled_imgs)

        labeled_preds = disc_model(mu)
        unlabeled_preds = disc_model(unlab_mu)

        lab_real_preds = torch.ones(labeled_imgs.size(0), 1).cuda()
        unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0), 1).cuda()

        dsc_loss = bce_loss(labeled_preds, lab_real_preds) + bce_loss(
            unlabeled_preds, unlab_fake_preds
        )

        optim_disc.zero_grad()
        dsc_loss.backward()
        optim_disc.step()

        # if temp_iter % 5 == 0:
        #     return

        # print("Epoch[{}],Iteration [{}/{}], VAE Loss: {:.3f}, Disc Loss: {:.4f}"\
        #         .format(cur_epoch,temp_iter, train_iterations, total_vae_loss.item(), dsc_loss.item()))

        print(
            "Epoch[{}],Iteration [{}/{}], VAE Loss: {:.3f}, Disc Loss: {:.4f}".format(
                cur_epoch,
                temp_iter,
                train_iterations,
                total_vae_loss.item(),
                dsc_loss.item(),
            )
        )

        # if temp_iter%100 == 0:
        #     print("Epoch[{}],Iteration [{}/{}], VAE Loss: {:.3f}, Disc Loss: {:.4f}"\
        #         .format(cur_epoch,temp_iter, train_iterations, total_vae_loss.item(), dsc_loss.item()))

    return vae_model, disc_model, optim_vae, optim_disc, curr_vae_disc_iter


def train_vae_epoch(
    cfg, vae_model, vae_optim, luSetLoader, n_luSetPts, cur_epoch, isDistributed=False
):
    """Trains VAE for an epoch."""
    temp_bs = int(cfg.VAAL.VAE_BS)
    max_iters = 1 * int(n_luSetPts / temp_bs)
    print("Training VAE for {} iterations".format(max_iters))

    if isDistributed:
        luSetLoader.sampler.set_epoch(cur_epoch)

    data = read_data(luSetLoader, labels=False)
    vae_model.train()

    # for it in range(max_iters):
    for it in tqdm(range(max_iters), desc="Epoch " + str(cur_epoch)):
        x = next(data)
        x = x.type(torch.cuda.FloatTensor)
        x = x.cuda()

        recon, z, mu, logvar = vae_model(x)

        temp_vae_loss = vae_loss(x, recon, mu, logvar, cfg.VAAL.BETA)

        vae_optim.zero_grad()
        temp_vae_loss.backward()
        vae_optim.step()

        # if it is not 0 and it%5 == 0:
        #     return
        # print("Epoch[{}], Iteration [{}/{}], VAE_Loss: {}".format(cur_epoch,it, max_iters, temp_vae_loss.item()))

        # if it%500 == 0:
        #     print("Epoch[{}], Iteration [{}/{}], VAE_Loss: {}".format(cur_epoch,it, max_iters, temp_vae_loss.item()))

    return vae_model, vae_optim


def train_vae_disc(cfg, lSet, uSet, trainDataset, dataObj, debug=False):
    """Trains VAE and discriminator to facilitate VAAL sampling.

    Args:
        cfg : Reference to config yaml.
        lSet (np.ndarray): Labeled set.
        uSet (np.ndarray): Unlabeled set.
        trainDataset (torch.utils.data.Dataset): Reference to the training dataset.
        dataObj: Reference to data class.

    Returns:
        torch.nn.Module, torch.nn.Module: Returns VAE and discriminator.
    """

    cur_device = torch.cuda.current_device()

    if cfg.TRAIN.DATASET == "MNIST":
        vae_model = vm.MNIST_VAE(cur_device, z_dim=cfg.VAAL.Z_DIM, nc=1)
    else:
        vae_model = vm.VAE(cur_device, z_dim=cfg.VAAL.Z_DIM)

    disc_model = vm.Discriminator(z_dim=cfg.VAAL.Z_DIM)

    # vae_model = distributed_wrapper(vae_model, cur_device)
    # disc_model = distributed_wrapper(disc_model, cur_device)

    vae_model = data_parallel_wrapper(vae_model, cur_device, cfg)
    disc_model = data_parallel_wrapper(disc_model, cur_device, cfg)

    # lSetLoader = dataObj.getDistributedIndexesDataLoader(indexes=lSet, batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS) \
    #         ,data=trainDataset, n_worker=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=cfg.DATA_LOADER.PIN_MEMORY, drop_last=True)

    # uSetLoader = dataObj.getDistributedIndexesDataLoader(indexes=uSet, batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS) \
    #         ,data=trainDataset, n_worker=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=cfg.DATA_LOADER.PIN_MEMORY, drop_last=True)

    if cfg.TRAIN.DATASET == "IMAGENET":
        lSetLoader = imagenet_loader.construct_loader_no_aug(
            cfg, indices=lSet, isDistributed=False, isVaalSampling=True
        )
        uSetLoader = imagenet_loader.construct_loader_no_aug(
            cfg, indices=uSet, isDistributed=False, isVaalSampling=True
        )
    else:
        print("=======================================")
        print(f"Data loader batch size: {int(cfg.VAAL.VAE_BS)}")
        print("=======================================")
        lSetLoader = dataObj.getIndexesDataLoader(
            indexes=lSet, batch_size=int(cfg.VAAL.VAE_BS), data=trainDataset
        )

        uSetLoader = dataObj.getIndexesDataLoader(
            indexes=uSet, batch_size=int(cfg.VAAL.VAE_BS), data=trainDataset
        )

    print("Initializing VAE and discriminator")
    optim_vae = torch.optim.Adam(vae_model.parameters(), lr=cfg.VAAL.VAE_LR)
    print(f"VAE Optimizer ==> {optim_vae}")
    optim_disc = torch.optim.Adam(disc_model.parameters(), lr=cfg.VAAL.DISC_LR)
    print(f"Disc Optimizer ==> {optim_disc}")
    print("==================================")

    n_lu_points = len(lSet) + len(uSet)
    max_vae_disc_iters = int(n_lu_points / cfg.VAAL.VAE_BS) * cfg.VAAL.VAE_EPOCHS
    change_lr_iter = max_vae_disc_iters // 25
    curr_vae_disc_iter = 0
    for epoch in range(cfg.VAAL.VAE_EPOCHS):
        (
            vae_model,
            disc_model,
            optim_vae,
            optim_disc,
            curr_vae_disc_iter,
        ) = train_vae_disc_epoch(
            cfg,
            vae_model,
            disc_model,
            optim_vae,
            optim_disc,
            lSetLoader,
            uSetLoader,
            epoch,
            n_lu_points,
            curr_vae_disc_iter,
            max_vae_disc_iters,
            change_lr_iter,
        )
        # print(f"VAE_DISC ITER COUNT PROGRESS: [{curr_vae_disc_iter}/{max_vae_disc_iters}]")

    # Save vae and disc models
    vae_sd = (
        vae_model.module.state_dict() if cfg.NUM_GPUS > 1 else vae_model.state_dict()
    )
    disc_sd = (
        disc_model.module.state_dict() if cfg.NUM_GPUS > 1 else disc_model.state_dict()
    )
    # Record the state
    vae_checkpoint = {
        "epoch": cfg.VAAL.VAE_EPOCHS + 1,
        "model_state": vae_sd,
        "optimizer_state": optim_vae.state_dict(),
        "cfg": cfg.dump(),
    }
    disc_checkpoint = {
        "epoch": cfg.VAAL.VAE_EPOCHS + 1,
        "model_state": disc_sd,
        "optimizer_state": optim_disc.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    vae_checkpoint_file = os.path.join(cfg.OUT_DIR, "vae.pyth")
    disc_checkpoint_file = os.path.join(cfg.OUT_DIR, "disc.pyth")
    torch.save(vae_checkpoint, vae_checkpoint_file)
    torch.save(disc_checkpoint, disc_checkpoint_file)

    if debug:
        print("Saved VAE model at {}".format(vae_checkpoint_file))
    if debug:
        print("Saved DISC model at {}".format(disc_checkpoint_file))

    return vae_model, disc_model


def train_vae(cfg, lSet, uSet, trainDataset, dataObj, debug=False):

    cur_device = 0  # torch.cuda.current_device()
    if cfg.TRAIN.DATASET == "MNIST":
        vae_model = vm.MNIST_VAE(cur_device, z_dim=cfg.VAAL.Z_DIM, nc=1)
    else:
        vae_model = vm.VAE(cur_device, z_dim=cfg.VAAL.Z_DIM)
    # vae_model = distributed_wrapper(cfg, vae_model, cur_device)

    vae_model = data_parallel_wrapper(vae_model, cur_device, cfg)

    luSet = np.append(lSet, uSet)
    n_luSetPts = len(luSet)

    # luSetLoader = dataObj.getDistributedIndexesDataLoader(indexes=luSet, batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS) \
    #         ,data=trainDataset, n_worker=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=cfg.DATA_LOADER.PIN_MEMORY, drop_last=True)

    if cfg.TRAIN.DATASET == "IMAGENET":
        luSetLoader = imagenet_loader.construct_loader_no_aug(
            cfg, indices=luSet, isDistributed=False, isVaalSampling=True
        )
    else:
        luSetLoader = dataObj.getIndexesDataLoader(
            indexes=luSet, batch_size=int(cfg.VAAL.VAE_BS), data=trainDataset
        )

    optim_vae = torch.optim.Adam(vae_model.parameters(), lr=cfg.VAAL.VAE_LR)
    print("=== FOR TRAINING VAE IN VAAL[-D] ===")
    print(f"optimizer: {optim_vae}")
    print("====================================")

    for epoch in range(cfg.VAAL.VAE_EPOCHS):
        vae_model, optim_vae = train_vae_epoch(
            cfg, vae_model, optim_vae, luSetLoader, n_luSetPts, epoch
        )

    # Save vae model
    vae_sd = (
        vae_model.module.state_dict() if cfg.NUM_GPUS > 1 else vae_model.state_dict()
    )

    # Record the state
    vae_checkpoint = {
        "epoch": cfg.VAAL.VAE_EPOCHS + 1,
        "model_state": vae_sd,
        "optimizer_state": optim_vae.state_dict(),
        "cfg": cfg.dump(),
    }

    # Write the checkpoint
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    vae_checkpoint_file = os.path.join(cfg.OUT_DIR, "vae.pyth")
    torch.save(vae_checkpoint, vae_checkpoint_file)

    if debug:
        print("Saved VAE model at {}".format(vae_checkpoint_file))

    return vae_model


def train_vaal(val_acc, val_epoch, trainDataset, valDataset, dataObj):
    """Implements VAAl sampling utility"""

    # We pass valDataset because it contains no transformation
    lSet, uSet, valSet = dataObj.loadPartitions(
        lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH,
        uSetPath=cfg.ACTIVE_LEARNING.USET_PATH,
        valSetPath=cfg.ACTIVE_LEARNING.VALSET_PATH,
    )
    print("====== Partitions Loaded =======")
    print("lSet: {}, uSet:{}, valSet: {}".format(len(lSet), len(uSet), len(valSet)))
    print("================================")

    vae, disc = train_vae_disc(lSet, uSet, valDataset, dataObj)

    # do active sampling
