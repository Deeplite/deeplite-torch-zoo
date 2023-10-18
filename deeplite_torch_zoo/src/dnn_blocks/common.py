# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Code taken from:
# - https://github.com/ultralytics/yolov5/
# The file is modified by Deeplite Inc. from the original implementation on Mar 21, 2023
# Code refactoring

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mish_cuda import MishCuda as Mish
except:

    class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
        def forward(self, x):
            return x * torch.nn.functional.softplus(x).tanh()


from deeplite_torch_zoo.src.registries import VARIABLE_CHANNEL_BLOCKS
from deeplite_torch_zoo.utils import LOGGER


ACT_TYPE_MAP = {
    'relu': nn.ReLU(inplace=True),
    'relu6': nn.ReLU6(inplace=True),
    'hswish': nn.Hardswish(inplace=True),
    'hardswish': nn.Hardswish(inplace=True),
    'silu': nn.SiLU(inplace=True),
    'lrelu': nn.LeakyReLU(0.1, inplace=True),
    'hsigmoid': nn.Hardsigmoid(inplace=True),
    'sigmoid': nn.Sigmoid(),
    'mish': Mish(),
    'leakyrelu': nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'leakyrelu_0.1': nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'gelu': nn.GELU(),
}


def get_activation(activation_name):
    if activation_name:
        return ACT_TYPE_MAP[activation_name]
    LOGGER.debug('No activation specified for get_activation. Returning nn.Identity()')
    return nn.Identity()


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def round_channels(channels, divisor=8):
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


@VARIABLE_CHANNEL_BLOCKS.register()
class ConvBnAct(nn.Module):
    # Standard convolution-batchnorm-activation block
    def __init__(
        self,
        c1,  # input channels
        c2,  # output channels
        k=1,  # kernel size
        s=1,  # stride
        p=None,  # padding
        g=1,  # groups
        b=None,  # bias
        act='relu',  # activation, either a string or a nn.Module; nn.Identity if None
        d=1,  # dilation
        residual=False,  # whether do add a skip connection
        use_bn=True,  # whether to use BatchNorm
        channel_divisor=1,  # round the number of out channels to the nearest multiple of channel_divisor
    ):
        super().__init__()

        # YOLOv5 applies channel_divisor=8 by default
        c2 = round_channels(c2, channel_divisor)

        self.in_channels = c1
        self.out_channels = c2
        self.use_bn = use_bn
        b = not self.use_bn if b is None else b

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d, groups=g, bias=b)

        self.bn = nn.BatchNorm2d(c2) if use_bn else nn.Identity()
        self.act = ACT_TYPE_MAP[act] if act else nn.Identity()
        self.residual = residual

        self.resize_identity = (c1 != c2) or (s != 1)

        if self.residual:
            # in case the input and output shapes are different, we need a 1x1 conv in the skip connection
            self.identity_conv = nn.Sequential()
            self.identity_conv.add_module(
                'conv', nn.Conv2d(c1, c2, 1, s, autopad(1, p), bias=b)
            )
            if self.use_bn:
                self.identity_conv.add_module('bn', nn.BatchNorm2d(c2))

    def forward(self, x):
        inp = x
        out = self.act(self.bn(self.conv(x)))
        if self.residual:
            if self.resize_identity:
                out = out + self.identity_conv(inp)
            else:
                out = out + inp
        return out

    def forward_fuse(self, x):
        inp = x
        out = self.act(self.conv(x))
        if self.residual:
            if self.resize_identity:
                out = out + self.identity_conv(inp)
            else:
                out = out + inp
        return out


@VARIABLE_CHANNEL_BLOCKS.register()
class DWConv(ConvBnAct):
    # Depth-wise convolution class
    def __init__(
        self, c1, c2, k, s=1, act='relu', residual=False, use_bn=True, channel_divisor=1
    ):  # ch_in, kernel, stride, padding, groups
        if c1 != c2:
            raise ValueError('Input and output channel count of DWConv does not match')
        super().__init__(
            c1,
            c2,
            k,
            s,
            g=c1,
            act=act,
            residual=residual,
            use_bn=use_bn,
            channel_divisor=channel_divisor,
        )


@VARIABLE_CHANNEL_BLOCKS.register()
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        g=1,
        dw_k=5,
        dw_s=1,
        act='relu',
        shrink_factor=0.5,
        residual=False,
    ):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels

        self.residual = residual
        self.single_conv = False
        if c_ < 2:
            self.single_conv = True
            self.cv1 = ConvBnAct(c1, c2, k, s, p=None, g=g, act=act)
        else:
            self.cv1 = ConvBnAct(c1, c_, k, s, p=None, g=g, act=act)
            self.cv2 = ConvBnAct(c_, c_, dw_k, dw_s, p=None, g=c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        if self.single_conv:
            return y
        return (
            torch.cat([y, self.cv2(y)], 1)
            if not self.residual
            else x + torch.cat([y, self.cv2(y)], 1)
        )


@VARIABLE_CHANNEL_BLOCKS.register()
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act='relu'
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = ConvBnAct(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat(
                (
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ),
                1,
            )
        )


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


@VARIABLE_CHANNEL_BLOCKS.register()
class ConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, x):
        return super().forward(x)
