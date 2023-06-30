# 2020.11.06-Changed for building GhostNetV2
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Creates a GhostNet Model as defined in:
# GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
# https://arxiv.org/abs/1911.11907
# Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models

# Taken from https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
# The file is modified by Deeplite Inc. from the original implementation on Jan 18, 2023
# Code implementation refactoring

import torch.nn as nn
import torch.nn.functional as F

from deeplite_torch_zoo.src.dnn_blocks.common import (
    round_channels,
    get_activation,
    ConvBnAct,
)
from deeplite_torch_zoo.src.dnn_blocks.ghostnetv2.ghostnet_blocks import (
    GhostBottleneckV2,
)


class GhostNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, act='relu'):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = round_channels(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = get_activation(act)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = round_channels(c * width, 4)
                hidden_channel = round_channels(exp_size * width, 4)
                layers.append(
                    GhostBottleneckV2(
                        input_channel,
                        output_channel,
                        hidden_channel,
                        k,
                        s,
                        se_ratio=se_ratio,
                        layer_id=layer_id,
                        act=act,
                    )
                )
                input_channel = output_channel
                layer_id += 1
            stages.append(nn.Sequential(*layers))

        output_channel = round_channels(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = get_activation(act)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet_v2(num_classes=1000, width=1.0, dropout=0.0, act='relu'):
    cfgs = [
        # k, t, c, SE, s
        [[3, 16, 16, 0, 1]],
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        [[3, 240, 80, 0, 2]],
        [
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1],
        ],
        [[5, 672, 160, 0.25, 2]],
        [
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
        ],
    ]
    return GhostNetV2(
        cfgs,
        num_classes=num_classes,
        width=width,
        dropout=dropout,
        act=act,
    )
