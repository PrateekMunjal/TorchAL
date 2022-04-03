def get_resnet_style_model(cfg):

    if cfg.TRAIN.TRANSFER_EXP:
        #Transfer experiment
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
            model = resnet.wide_resnet50_2(pretrained=False, num_classes=cfg.MODEL.NUM_CLASSES)
        else:
            raise NotImplementedError
    elif model_type == "wide_resnet_28_10":
        if model_depth == 28:
            model = wide_resnet.Wide_ResNet(depth=model_depth, widen_factor=10, \
                dropout_rate=0.3, num_classes=cfg.MODEL.NUM_CLASSES)
        else:
            raise NotImplementedError
    elif model_type == "wide_resnet_28_2":
        if model_depth == 28:
            model = wide_resnet.Wide_ResNet(depth=model_depth, widen_factor=2, \
                dropout_rate=0.3, num_classes=cfg.MODEL.NUM_CLASSES)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return model