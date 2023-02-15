import torch
import torch.nn as nn
from deeplite_torch_zoo.src.dnn_blocks.common import (ChannelShuffle,
                                                      ConvBnAct, DWConv,
                                                      get_activation)


class ShuffleUnit(nn.Module):
    """
    ShuffleNet unit.
    Original paper: 'ShuffleNet: An Extremely Efficient Convolutional Neural Network
    for Mobile Devices,https://arxiv.org/abs/1707.01083.

    Parameters:
    ----------
    c1 : int
        Number of input channels.
    c2 : int
        Number of output channels.
    g : int
        Number of groups in convolution layers.
    downsample : bool
        Whether do downsample.
    ignore_group : bool
        Whether ignore group value in the first convolution layer.
    """
    def __init__(self,
                 c1,
                 c2,
                 g=1,
                 e=4,
                 dw_k=3,
                 act='relu',
                 downsample=False,
                 ignore_group=False):
        super(ShuffleUnit, self).__init__()
        self.downsample = downsample
        mid_channels = c2 // e

        if downsample:
            c2 -= c1

        if c1 != c2:
            raise ValueError('Input and output channel count of ShuffleUnit does not match')

        self.compress_conv_bn_act1 = ConvBnAct(
            c1=c1,
            c2=mid_channels,
            k=1,
            g=(1 if ignore_group else g),
            act=act,
        )

        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=g)

        self.dw_conv_bn2 = DWConv(
            c1=mid_channels,
            c2=mid_channels,
            k=dw_k,
            s=(2 if self.downsample else 1),
            act=False)

        self.expand_conv_bn3 = ConvBnAct(
            c1=mid_channels,
            c2=c2,
            k=1,
            g=g,
            act=False)

        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activ = get_activation(act)

    def forward(self, x):
        identity = x
        x = self.compress_conv_bn_act1(x)
        x = self.c_shuffle(x)
        x = self.dw_conv_bn2(x)
        x = self.expand_conv_bn3(x)
        if self.downsample:
            identity = self.avgpool(identity)
            x = torch.cat((x, identity), dim=1)
        else:
            x = x + identity
        x = self.activ(x)
        return x


def _test_blocks(c1, c2, b=2, res=32):
    # perform a basic forward pass
    input = torch.rand((b,c1,res, res), device=None, requires_grad=False)

    fire_block = ShuffleUnit(c1, c2)
    output = fire_block(input)
    assert output.shape == (b, c2, res, res)


if __name__ == "__main__":
    _test_blocks(32, 32) #  c1 == c2
