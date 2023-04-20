# Taken from https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
# The file is modified by deeplite from the original implementation on Jan 18, 2023
# Code implementation refactoring

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplite_torch_zoo.src.dnn_blocks.common import (get_activation,
                                                      round_channels)
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.cnn_attention import SELayer


class DFCModule(nn.Module):
    def __init__(self, c1, c2, k, s, dfc_k=5, downscale=True):
        super().__init__()
        self.downscale = downscale
        self.gate_fn = nn.Sigmoid()
        self.short_conv = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(
                    c2,
                    c2,
                    kernel_size=(1, dfc_k),
                    stride=1,
                    padding=(0, 2),
                    groups=c2,
                    bias=False,
                ),
                nn.BatchNorm2d(c2),
                nn.Conv2d(
                    c2,
                    c2,
                    kernel_size=(dfc_k, 1),
                    stride=1,
                    padding=(2, 0),
                    groups=c2,
                    bias=False,
                ),
                nn.BatchNorm2d(c2),
            )


    def forward(self, x):
        res = F.avg_pool2d(x, kernel_size=2, stride=2) if self.downscale else x
        res = self.short_conv(res)
        res = self.gate_fn(res)
        if self.downscale:
            res = F.interpolate(res, size=x.shape[-1], mode='nearest')
        return res


class GhostModuleV2(nn.Module):
    def __init__(self, c1, c2, k=3, ratio=2, dw_size=3, s=1, act='relu', mode=None):
        super().__init__()
        self.mode = mode
        self.activation = get_activation(act)
        self.oup = c2
        init_channels = math.ceil(c2 / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, init_channels, k, s, k // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            get_activation(act),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            get_activation(act),
        )
        if self.mode == 'attn':
            self.dfc = DFCModule(c1, c2, k, s)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        res = out[:, : self.oup, :, :]
        if self.mode == 'attn':
            res = res * self.dfc(x)
        return res


class GhostBottleneckV2(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        mid_chs=None,
        e=2,
        dw_kernel_size=3,
        s=1,
        act='relu',
        se_ratio=0.0,
        use_attn=True,
    ):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = s
        if mid_chs is None:
            mid_chs = round_channels(c1 * e, divisor=4)

        # Point-wise expansion
        mode = 'attn' if use_attn else 'original'
        self.ghost1 = GhostModuleV2(c1, mid_chs, act=act, mode=mode)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=s,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        self.se = SELayer(mid_chs, reduction=1/se_ratio) if has_se else nn.Identity()
        self.ghost2 = GhostModuleV2(mid_chs, c2, act=False, mode='original')

        # shortcut
        if c1 == c2 and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    c1,
                    c1,
                    dw_kernel_size,
                    stride=s,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=c1,
                    bias=False,
                ),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c2, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c2),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


def _test_blocks(c1, c2, b=2, res=32):
    input = torch.rand((b, c1, res, res), device=None, requires_grad=False)

    block = GhostModuleV2(c1, c2, mode="original")
    output = block(input)
    assert output.shape == (b, c2, res, res)

    block = GhostBottleneckV2(c1, c2, 1, use_attn=True)
    output = block(input)
    assert output.shape == (b, c2, res, res)

    block = GhostBottleneckV2(c1, c2, 1, use_attn=False)
    output = block(input)
    assert output.shape == (b, c2, res, res)

    block = GhostModuleV2(c1, c2, mode="attn")
    output = block(input)
    assert output.shape == (b, c2, res, res)


if __name__ == "__main__":
    _test_blocks(64, 64)  #  c1 == c2
    _test_blocks(64, 32)  #  c1 > c2
    _test_blocks(32, 64)  #  c1 < c2
