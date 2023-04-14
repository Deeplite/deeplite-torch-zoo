# Modified from https://github.com/moskomule/senet.pytorch

import torch.nn as nn
from torch import Tensor

from deeplite_torch_zoo.src.dnn_blocks.cnn_attention import SELayer
from deeplite_torch_zoo.src.dnn_blocks.common import (ConvBnAct, DWConv,
                                                      GhostConv,
                                                      get_activation,
                                                      round_channels)


class ResNetBottleneck(nn.Module):
    # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py
    def __init__(
        self,
        c1: int,
        c2: int,
        e: float = 1.0,
        k: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        act='relu',
        se_ratio=None,
        channel_divisor=1,
        mid_channels=None,
        shortcut=True,
    ) -> None:
        super().__init__()

        self.resize_identity = (c1 != c2) or (stride != 1)
        c_ = mid_channels if mid_channels is not None else round_channels(e * c1, channel_divisor)
        self.shortcut = shortcut

        self.conv1 = ConvBnAct(c1, c_, 1, act=act)
        self.conv2 = ConvBnAct(c_, c_, k, stride, g=groups, d=dilation, act=act)
        self.conv3 = ConvBnAct(c_, c2, 1, act=False)
        self.act = get_activation(act)
        self.se = SELayer(c2, reduction=se_ratio) if se_ratio else nn.Identity()
        if self.resize_identity:
            self.identity_conv = ConvBnAct(c1, c2, 1, stride, act=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.shortcut:
            if self.resize_identity:
                identity = self.identity_conv(x)
            else:
                identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)

        if self.shortcut:
            out += identity

        out = self.act(out)

        return out


class ResNetBasicBlock(nn.Module):
    # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        stride: int = 1,
        act='relu',
        se_ratio=None,
        channel_divisor=1,
    ) -> None:
        super().__init__()
        c2 = round_channels(c2, channel_divisor)

        self.resize_identity = (c1 != c2) or (stride != 1)
        self.conv1 = ConvBnAct(c1, c2, k, stride, act=act)
        self.conv2 = ConvBnAct(c2, c2, k, act=False)
        self.act = get_activation(act)
        self.se = SELayer(c2, reduction=se_ratio) if se_ratio else nn.Identity()
        self.identity_conv = ConvBnAct(c1, c2, 1, stride, act=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out += identity
        out = self.act(out)

        return out


class ResNeXtBottleneck(ResNetBottleneck):
    def __init__(
        self,
        c1: int,
        c2: int,
        e: float = 1.0,
        k: int = 3,
        stride: int = 1,
        groups: int = 32,
        dilation: int = 1,
        act='relu',
        se_ratio=None,
    ):
        super(ResNeXtBottleneck, self).__init__(
            c1, c2, e, k, stride, groups,
            dilation, act, se_ratio, channel_divisor=groups)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck as described in https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1, shrink_factor=2):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1, shrink_factor=shrink_factor),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False, shrink_factor=shrink_factor))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      ConvBnAct(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
