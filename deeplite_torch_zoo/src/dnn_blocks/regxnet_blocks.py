""" ReXNet

A PyTorch impl of `ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network` -
https://arxiv.org/abs/2007.00992

Adapted from original impl at https://github.com/clovaai/rexnet
Copyright (c) 2020-present NAVER Corp. MIT license

Changes for timm, feature extraction, and rounded channel variant hacked together by Ross Wightman
Copyright 2020 Ross Wightman
"""

# Modified from https://github1s.com/huggingface/pytorch-image-models/blob/HEAD/timm/models/rexnet.py

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.cnn_attention import SEWithNorm
from deeplite_torch_zoo.src.dnn_blocks.common import (ConvBnAct,
                                                      get_activation,
                                                      round_channels)


class RexNetBottleneck(nn.Module):
    def __init__(self, c1, c2, stride=1, exp_ratio=1.0, k=3, se_ratio=None, ch_div=1,
            act='hswish', dw_act='relu6'):
        super(RexNetBottleneck, self).__init__()

        self.in_channels = c1
        self.out_channels = c2

        if exp_ratio != 1.:
            dw_chs = round_channels(round(c1 * exp_ratio), divisor=ch_div)
            self.conv_exp = ConvBnAct(c1, dw_chs, act=act)
        else:
            dw_chs = c1
            self.conv_exp = None

        self.conv_dw = ConvBnAct(dw_chs, dw_chs, k, s=stride, g=dw_chs, act=None)

        self.se = SEWithNorm(dw_chs, mid_channels=round_channels(int(dw_chs * se_ratio), ch_div)) \
            if se_ratio is not None else nn.Identity()

        self.act_dw = get_activation(dw_act)

        self.conv_pwl = ConvBnAct(dw_chs, c2, 1, act=None)

    def forward(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        x = torch.cat([x[:, 0:self.in_channels] + shortcut,
            x[:, self.in_channels:]], dim=1)
        return x
