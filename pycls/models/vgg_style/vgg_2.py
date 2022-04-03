"""
Modified from https://github.com/pytorch/vision.git
"""
import math

import torch.nn as nn
import torch.nn.init as init

# fmt: off
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]
# fmt: on


class VGG(nn.Module):
    """
    VGG model
    """

    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features

        self.num_classes = num_classes
        self.penultimate_active = False
        if self.num_classes == 1000:
            logger.warning(
                "This open source implementation is only suitable for small datasets like CIFAR. For Imagenet we recommend to use Resnet based models"
            )

        self.classifier_penultimate = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
        )

        self.final_classifier = nn.Linear(512, self.num_classes)

        # Describe model with source code link
        self.description = "Open Source Implementation of VGG16 adapted from chengyangfu/pytorch-vgg-cifar10 repository"

        self.source_link = (
            "https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py"
        )

        # Initialize weights

        ## This is Kaiming He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        z = self.classifier_penultimate(x)
        x = self.final_classifier(z)
        if self.penultimate_active:
            return z, x
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# fmt: off
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}
# fmt:on


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg["A"]))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg["A"], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg["B"]))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg["B"], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg["D"]))


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg["D"], batch_norm=True), **kwargs)


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg["E"]))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg["E"], batch_norm=True))
