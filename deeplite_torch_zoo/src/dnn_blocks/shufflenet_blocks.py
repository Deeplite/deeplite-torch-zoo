import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import (ConvBnAct, ACT_TYPE_MAP, conv1x1, DWConv, conv1x1)


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
                 act="relu",
                 downsample=False,
                 ignore_group=False):
        super(ShuffleUnit, self).__init__()
        self.downsample = downsample
        mid_channels = c2 // 4

        if downsample:
            c2 -= c1

        if c1 != c2:
            raise ValueError('Input and output channel count of ShuffleUnit does not match')

        self.compress_conv_bn1 = conv1x1(
            c1=c1,
            c2=mid_channels,
            g=(1 if ignore_group else g),
            use_bn=True,
            act=act)
        self.c_shuffle = ChannelShuffle(
            channels=mid_channels,
            groups=g)
        self.dw_conv_bn2 = DWConv(
            c1=mid_channels,
            c2=mid_channels,
            k=3,
            s=(2 if self.downsample else 1),
            act=act)
        self.expand_conv_bn3 = conv1x1(
            c1=mid_channels,
            c2=c2,
            g=g,
            use_bn=True,
            act=act)
        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activ = ACT_TYPE_MAP[act] if act else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.compress_conv_bn1(x)
        x = self.activ(x)
        x = self.c_shuffle(x)
        x  =self.dw_conv_bn2(x)
        x = self.expand_conv_bn3(x)
        if self.downsample:
            identity = self.avgpool(identity)
            x = torch.cat((x, identity), dim=1)
        else:
            x = x + identity
        x = self.activ(x)
        return x


def channel_shuffle(x, groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural 
    Network for Mobile Devices,' https://arxiv.org/abs/1707.01083.
    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.
    Returns:
    -------
    Tensor
        Resulted tensor.
    """
    batch, channels, height, width = x.size()
    # assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels,
                 groups):
        super(ChannelShuffle, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)

    def __repr__(self):
        s = "{name}(groups={groups})"
        return s.format(
            name=self.__class__.__name__,
            groups=self.groups)


def _test_blocks(c1, c2, b=2, res=32):
    # perform a basic forward pass
    input = torch.rand((b,c1,res, res), device=None, requires_grad=False)

    fire_block = ShuffleUnit(c1, c2)
    output = fire_block(input)
    assert output.shape == (b, c2, res, res)


if __name__ == "__main__":
    _test_blocks(32, 32) #  c1 == c2

