#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model construction functions."""

import torch

from pycls.models.anynet import AnyNet
from pycls.models.effnet import EffNet

# from pycls.models.resnet import ResNet

from pycls.models.vgg_style import vgg
from pycls.models.vgg_style import vgg_1
from pycls.models.vgg_style import vgg_2
from pycls.models.vgg_style import vgg_check
from pycls.models.vgg_style import vgg_mnist

from pycls.models.resnet_style import resnet
from pycls.models.resnet_style import resnet_1
from pycls.models.resnet_style import resnet_2
from pycls.models.resnet_style import resnet_shake_shake
from pycls.models.resnet_style import wide_resnet

import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

# Supported models
_models = {
    "anynet": AnyNet,
    "effnet": EffNet,
    #'resnet': ResNet,
    # VGG style architectures
    "vgg": vgg,
    "vgg_1": None,
    "vgg_2": None,
    "vgg_check": None,
    "vgg_mnist": None,
    # Resnet style architectures
    "resnet": None,
    "resnet_1": None,
    "resnet_2": None,
    "resnet18": None,
    "resnet50": None,
    "resnet_shake_shake": None,
    "wide_resnet_28_10": None,
    "wide_resnet_28_2": None,
    "wide_resnet_50": None,
}


def get_resnet_style_model(cfg):

    if cfg.TRAIN.TRANSFER_EXP:
        # Transfer experiment
        model_type = cfg.MODEL.TRANSFER_MODEL_TYPE
        model_depth = cfg.MODEL.TRANSFER_MODEL_DEPTH
    else:
        model_type = cfg.MODEL.TYPE
        model_depth = cfg.MODEL.DEPTH

    if model_type == "resnet":
        if model_depth == 50:
            model = resnet.resnet50(pretrained=False, num_classes=cfg.MODEL.NUM_CLASSES)
        elif model_depth == 18:
            model = resnet.resnet18(pretrained=False, num_classes=cfg.MODEL.NUM_CLASSES)
        else:
            raise NotImplementedError
    elif model_type == "resnet_1":
        model = resnet_1.ResNet(temp_cfg=cfg)

    elif model_type == "resnet_2":
        if model_depth == 50:
            model = resnet_2.ResNet50(num_classes=cfg.MODEL.NUM_CLASSES)
        elif model_depth == 18:
            model = resnet_2.ResNet18(num_classes=cfg.MODEL.NUM_CLASSES)
        else:
            raise NotImplementedError

    elif model_type == "resnet_shake_shake":
        model = resnet_shake_shake.Network(cfg=cfg)
    elif model_type == "wide_resnet_50":
        if model_depth == 50:
            model = resnet.wide_resnet50_2(
                pretrained=False, num_classes=cfg.MODEL.NUM_CLASSES
            )
        else:
            raise NotImplementedError
    elif model_type == "wide_resnet_28_10":
        if model_depth == 28:
            model = wide_resnet.Wide_ResNet(
                depth=model_depth,
                widen_factor=10,
                dropout_rate=0.3,
                num_classes=cfg.MODEL.NUM_CLASSES,
            )
        else:
            raise NotImplementedError
    elif model_type == "wide_resnet_28_2":
        if model_depth == 28:
            model = wide_resnet.Wide_ResNet(
                depth=model_depth,
                widen_factor=2,
                dropout_rate=0.3,
                num_classes=cfg.MODEL.NUM_CLASSES,
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return model


def get_vgg_style_model(cfg):

    if cfg.TRAIN.TRANSFER_EXP:
        # Transfer experiment
        model_type = cfg.MODEL.TRANSFER_MODEL_TYPE
        model_depth = cfg.MODEL.TRANSFER_MODEL_DEPTH
    else:
        model_type = cfg.MODEL.TYPE
        model_depth = cfg.MODEL.DEPTH

    if model_type == "vgg":
        model = vgg.vgg16_bn(num_classes=cfg.MODEL.NUM_CLASSES, seed_id=cfg.RNG_SEED)
    elif model_type == "vgg_mnist":
        model = vgg_mnist.vgg16_bn(
            num_classes=cfg.MODEL.NUM_CLASSES, seed_id=cfg.RNG_SEED
        )
    elif model_type == "vgg_1":
        model = vgg_1.VGG("VGG16", num_classes=cfg.MODEL.NUM_CLASSES)
    elif model_type == "vgg_2":
        model = vgg_2.vgg16_bn(num_classes=cfg.MODEL.NUM_CLASSES)
    elif model_type == "vgg_check":
        model = vgg_check.vgg16_bn(num_classes=cfg.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError

    return model


def build_model(cfg, active_sampling=False, isDistributed=True):

    if cfg.TRAIN.TRANSFER_EXP:
        # Transfer experiment
        model_type = cfg.MODEL.TRANSFER_MODEL_TYPE
        model_depth = cfg.MODEL.TRANSFER_MODEL_DEPTH
    else:
        model_type = cfg.MODEL.TYPE
        model_depth = cfg.MODEL.DEPTH

    """Builds the model."""
    assert model_type in _models.keys(), "Model type '{}' not supported".format(
        model_type
    )

    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    print(f"########### cfg model type: {model_type}")

    # Construct the model
    if model_type.startswith("vgg"):
        model = get_vgg_style_model(cfg)

    elif model_type.startswith("resnet") or model_type.startswith("wide_resnet"):
        # print("Inside build_model of model_builder.py")
        # print(f"cfg train dataset: {cfg.TRAIN.DATASET}")
        model = get_resnet_style_model(cfg)
    else:
        raise NotImplementedError

    logger.info("Model loaded successfully!!")
    # temp_attribs = list(model.__dict__.keys())
    # assert 'description' in temp_attribs, "Model class has no <description> named attribute"
    # logger.info("[Description]: {}".format(model.description))
    # assert 'source_link' in temp_attribs, "Model class has no <source_link> named attribute"
    # logger.info("[Source Link]: {}".format(model.source_link))

    if not isDistributed:
        # print("returning from if not in distributed")
        return model

    # Construct the model
    # if model_type == 'vgg':
    #     #model = _models[model_type](num_classes=cfg.MODEL.NUM_CLASSES)
    #     model = vgg.vgg16_bn(num_classes=cfg.MODEL.NUM_CLASSES)
    # elif model_type == "vgg_git":
    #     from pycls.models import vgg_git
    #     model = vgg_git.VGG('VGG16')
    # elif model_type == "resnet18":
    #     from torchvision.models import resnet18
    #     model = resnet18(pretrained=False, num_classes=cfg.MODEL.NUM_CLASSES)
    # elif model_type == "resnet50":
    #     from torchvision.models import resnet50
    #     model = resnet50(pretrained=False, num_classes=cfg.MODEL.NUM_CLASSES)
    # else:
    #     model = _models[model_type]()

    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()

    # print("============In build model=============")
    # print("current_model: {}".format(model))
    # print("current_device: {}".format(cur_device))
    # print("=======================================")

    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    if active_sampling and not isDistributed:
        return model
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:

        ##################
        ## CHECK THIS
        ##################

        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, cur_device)
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # print("===== Added Sync Batchnorm =====")

        # Make model replica operate on the current device
        if torch.__version__ == "1.8.1":
            model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[cur_device],
                output_device=cur_device,
                gradient_as_bucket_view=True,
            )
            print("-------------")
            print("model with grad as bucket view: True")
            print("-------------")
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device
            )
    else:
        pass
        # print("============================")
        # print("Loading in DataParallel")
        # print("============================")
        # model = torch.nn.DataParallel(model, device_ids = [cur_device])

    return model


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor
