# Taken from: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/common.py

from functools import partial

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import (get_activation,
                                                      round_channels)


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    mid_channels : int or None, default None
        Number of middle channels.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    activation : function, or str, or nn.Module, default 'relu'
        Activation function after the first convolution.
    out_activation : function, or str, or nn.Module, default 'sigmoid'
        Activation function after the last convolution.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 mid_channels=None,
                 round_mid=False,
                 use_conv=True,
                 mid_activation='relu',
                 out_activation='hsigmoid',
                 norm_layer=None):
        super(SELayer, self).__init__()
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = channels // reduction if not round_mid \
                else round_channels(float(channels) / reduction)

        mid_channels += 1

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=True)
        else:
            self.fc1 = nn.Linear(
                in_features=channels,
                out_features=mid_channels)

        self.bn = norm_layer(mid_channels) if norm_layer else nn.Identity()
        self.activ = get_activation(mid_activation)

        if use_conv:
            self.conv2 = nn.Conv2d(
                in_channels=mid_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=True)
        else:
            self.fc2 = nn.Linear(
                in_features=mid_channels,
                out_features=channels
            )
        self.sigmoid = get_activation(out_activation)

    def forward(self, x):
        w = self.pool(x)

        # timm:
        # w = x.mean((2, 3), keepdim=True)
        # if self.add_maxpool:
        # # experimental codepath, may remove or change
        #   w = 0.5 * w + 0.5 * x.amax((2, 3), keepdim=True)

        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(self.bn(w))
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x


SEWithNorm = partial(SELayer, norm_layer=nn.BatchNorm2d)
