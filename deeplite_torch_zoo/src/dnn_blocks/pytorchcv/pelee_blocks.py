# Taken from https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/peleenet.py

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct


class PeleeBranch1(nn.Module):
    """
    PeleeNet branch type 1 block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        kernel_size=3,
        stride=1,
        act='relu',
    ):
        super(PeleeBranch1, self).__init__()
        self.conv1 = ConvBnAct(
            c1=in_channels, c2=mid_channels, k=1, s=stride, p=0, act=act
        )
        self.conv2 = ConvBnAct(
            c1=mid_channels, c2=out_channels, k=kernel_size, s=stride, act=act
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PeleeBranch2(nn.Module):
    """
    PeleeNet branch type 2 block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        stride=1,
        kernel_size=3,
        act='relu',
    ):
        super(PeleeBranch2, self).__init__()
        self.conv1 = ConvBnAct(
            c1=in_channels, c2=mid_channels, p=0, k=1, s=stride, act=act
        )
        self.conv2 = ConvBnAct(
            c1=mid_channels, c2=out_channels, k=kernel_size, s=stride, act=act
        )
        self.conv3 = ConvBnAct(
            c1=out_channels, c2=out_channels, k=kernel_size, s=stride, act=act
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class TwoStackDenseBlock(nn.Module):
    """
    PeleeNet dense block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bottleneck_size : int
        Bottleneck width.
    """

    def __init__(
        self, c1, c2, expansion_factor=2, bottleneck_size=1, k1=3, k2=3, act='relu'
    ):
        super().__init__()
        inc_channels = (int(expansion_factor * c2) - c1) // 2
        mid_channels = int(inc_channels * bottleneck_size)

        self.branch1 = PeleeBranch1(
            in_channels=c1,
            out_channels=inc_channels,
            mid_channels=mid_channels,
            kernel_size=k1,
            act=act,
        )
        self.branch2 = PeleeBranch2(
            in_channels=c1,
            out_channels=inc_channels,
            mid_channels=mid_channels,
            kernel_size=k2,
            act=act,
        )
        self.conv = ConvBnAct(c1=int(expansion_factor * c2), c2=c2, k=1, act=act)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat((x, x1, x2), dim=1)
        x = self.conv(x)
        return x
