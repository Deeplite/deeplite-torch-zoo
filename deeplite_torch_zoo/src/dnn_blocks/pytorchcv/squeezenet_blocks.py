# Taken from:
# - https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/squeezenet.py
# - https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/squeezenext.py

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct, get_activation


class FireUnit(nn.Module):
    """
    SqueezeNet unit, so-called 'Fire' unit.
    Parameters:
    ----------
    c1 : int
        Number of input channels.
    c2 : int
        Number of output channels.
    e : float , default 1/8
        Number of internal channels for squeeze convolution blocks.
    act : string
        Activation function to be used
    residual : bool
        Whether use residual connection.
    """

    def __init__(
        self,
        c1,
        c2,
        k=3,
        e=0.125,
        act='relu',
        residual=True,
        use_bn=True,
    ):
        super(FireUnit, self).__init__()
        self.residual = residual
        in_channels = c1
        expand_channels = c2 // 2
        squeeze_channels = int(
            c2 * e
        )  # Number of output channels for squeeze conv blocks.
        expand1x1_channels = (
            expand_channels  # Number of output channels for expand 1x1 conv blocks.
        )
        expand3x3_channels = (
            expand_channels  # Number of output channels for expand 3x3 conv blocks.
        )

        self.squeeze = ConvBnAct(
            c1=in_channels, c2=squeeze_channels, k=1, act=act, use_bn=use_bn
        )

        self.expand1x1 = ConvBnAct(
            c1=squeeze_channels, c2=expand1x1_channels, k=1, act=act, use_bn=use_bn
        )

        self.expand3x3 = ConvBnAct(
            c1=squeeze_channels, c2=expand3x3_channels, k=k, p=1, act=act, use_bn=use_bn
        )

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.squeeze(x)
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        out = torch.cat((y1, y2), dim=1)
        if self.residual:
            out = out + identity
        return out


class SqnxtUnit(nn.Module):
    """
    SqueezeNext unit.
    Original paper: 'SqueezeNext: Hardware-Aware Neural Network Design,'
    https://arxiv.org/abs/1803.10615.

    Parameters:
    ----------
    c1 : int
        Number of input channels.
    c2 : int
        Number of output channels.
    s : int or tuple/list of 2 int
        Strides of the convolution.
    """

    def __init__(self, c1, c2, e=2, k=3, s=1, act='relu'):
        super(SqnxtUnit, self).__init__()

        if s == 2:
            reduction_den = 1
            self.resize_identity = True
        elif c1 > c2:
            reduction_den = e * 2
            self.resize_identity = True
        elif c1 <= c2:
            reduction_den = e
            self.resize_identity = False

        self.conv1 = ConvBnAct(
            c1=c1, c2=(c1 // reduction_den), k=1, s=s, b=True, act=act
        )

        self.conv2 = ConvBnAct(
            c1=(c1 // reduction_den),
            c2=(c1 // (2 * reduction_den)),
            k=1,
            b=True,
            act=act,
        )

        self.conv3 = ConvBnAct(
            c1=(c1 // (2 * reduction_den)),
            c2=(c1 // reduction_den),
            k=(1, k),
            s=1,
            p=(0, 1),
            act=act,
        )

        self.conv4 = ConvBnAct(
            c1=(c1 // reduction_den),
            c2=(c1 // reduction_den),
            k=(k, 1),
            s=1,
            p=(1, 0),
            act=act,
        )

        self.conv5 = ConvBnAct(c1=(c1 // reduction_den), c2=c2, k=1, b=True, act=act)

        if self.resize_identity:
            self.identity_conv = ConvBnAct(c1=c1, c2=c2, s=s, act=act)
        self.activ = get_activation(act)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + identity
        x = self.activ(x)
        return x
