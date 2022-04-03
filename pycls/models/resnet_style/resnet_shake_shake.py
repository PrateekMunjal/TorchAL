import torch
import torch.nn as nn
import torch.nn.functional as F

from pycls.models.resnet_style.shake_shake_function import get_alpha_beta, shake_function

import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class ResidualPath(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualPath, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.bn2(self.conv2(x))
        return x


class DownsamplingShortcut(nn.Module):
    def __init__(self, in_channels):
        super(DownsamplingShortcut, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(in_channels * 2)

    def forward(self, x):
        x = F.relu(x, inplace=False)
        y1 = F.avg_pool2d(x, kernel_size=1, stride=2, padding=0)
        y1 = self.conv1(y1)

        y2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        y2 = F.avg_pool2d(y2, kernel_size=1, stride=2, padding=0)
        y2 = self.conv2(y2)

        z = torch.cat([y1, y2], dim=1)
        z = self.bn(z)

        return z


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shake_config):
        super(BasicBlock, self).__init__()

        self.shake_config = shake_config

        self.residual_path1 = ResidualPath(in_channels, out_channels, stride)
        self.residual_path2 = ResidualPath(in_channels, out_channels, stride)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('downsample',
                                     DownsamplingShortcut(in_channels))

    def forward(self, x):
        x1 = self.residual_path1(x)
        x2 = self.residual_path2(x)

        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = (False, False, False)

        alpha, beta = get_alpha_beta(x.size(0), shake_config, x.device)
        y = shake_function(x1, x2, alpha, beta)

        return self.shortcut(x) + y


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__() 

        print("Building Network of resnet Shake-Shake")
        
        self.cfg = cfg

        input_shape = (1, self.cfg.TRAIN.IM_CHANNELS, self.cfg.TRAIN.IM_SIZE, \
             self.cfg.TRAIN.IM_SIZE)
        
        print(f"Input_shape: {input_shape}")
        # input_shape = (1,3,224,224)

        n_classes = self.cfg.MODEL.NUM_CLASSES

        base_channels = self.cfg.SHAKE_SHAKE.BASE_CHANNELS
        #config['base_channels']
        depth = self.cfg.SHAKE_SHAKE.DEPTH
        #config['depth']
        self.shake_config = (self.cfg.SHAKE_SHAKE.FORWARD, self.cfg.SHAKE_SHAKE.BACKWARD, self.cfg.SHAKE_SHAKE.IMAGE)
        #(config['shake_forward'], config['shake_backward'],config['shake_image'])
        
        block = BasicBlock
        
        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth, f"Condition (n_blocks_per_stage * 6 + 2) == model_depth fails as model_depth: {model_depth} \
            and n_blocks_per_stage*6 + 2 = {n_blocks_per_stage*6 + 2}"

        n_channels = [base_channels, base_channels * 2, base_channels * 4]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.bn = nn.BatchNorm2d(base_channels)

        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage, block, stride=1)
        self.stage2 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage, block, stride=2)

        # # compute conv feature size

        # with torch.no_grad():
        #     self.feature_size = self._forward_conv(
        #         torch.zeros(1,3,224,224)).view(-1).shape[0]
        
        self.feature_size = 128

        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     self.feature_size = 128
        # else:
        #     self.feature_size = 128
        #     #raise NotImplementedError

        self.fc = nn.Linear(self.feature_size, n_classes)

        ### 
        self.penultimate_active = False
        #Describe model with source code link
        self.description = "Open source implementation of Resnet shake-shake adapted from hysts/pytorch_shake_shake/ repository"

        self.source_link = "https://github.com/hysts/pytorch_shake_shake/blob/master/shake_shake.py"

        self.model_depth = cfg.MODEL.TRANSFER_MODEL_DEPTH if cfg.TRAIN.TRANSFER_EXP else cfg.MODEL.DEPTH 

        logger.info('Constructing: Shake Shake ResNet with depth:{}'.format(self.model_depth))

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name,
                    block(
                        in_channels,
                        out_channels,
                        stride=stride,
                        shake_config=self.shake_config))
            else:
                stage.add_module(
                    block_name,
                    block(
                        out_channels,
                        out_channels,
                        stride=1,
                        shake_config=self.shake_config))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        z = x.view(x.size(0), -1)
        # print(f"~~z.shape[1]: {z.shape[1]}")
        x = self.fc(z)
        if self.penultimate_active:
            return z,x
        return x