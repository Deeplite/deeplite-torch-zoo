# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Code taken from:
# - https://github.com/ultralytics/yolov5/
# - https://github.com/osmr/imgclsmob
# - https://github.com/huggingface/pytorch-image-models/


import torch
import torch.nn as nn

try:
    from mish_cuda import MishCuda as Mish
except:
    class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
        def forward(self, x):
            return x * torch.nn.functional.softplus(x).tanh()


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
}


def get_activation(activation_name):
    return ACT_TYPE_MAP[activation_name] if activation_name else nn.Identity()


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def round_channels(channels, divisor=8):
    """
    Round weighted channel number (make divisible operation).
    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.
    Returns:
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


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
        out = self.act(self.bn(self.conv(x)))
        if self.residual:
            if self.resize_identity:
                out += self.identity_conv(x)
            else:
                out += x
        return out

    def forward_fuse(self, x):
        out = self.act(self.conv(x))
        if self.residual:
            if self.resize_identity:
                out += self.identity_conv(x)
            else:
                out += x
        return out


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


class GhostConv(nn.Module):
    # Ghost Convolution block https://github.com/huawei-noah/ghostnet
    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        g=1,
        act='relu',
        dw_k=3,
        dw_s=1,
        residual=False,
        shrink_factor=2,
    ):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = int(c2 / shrink_factor)  # hidden channels
        self.single_conv = False
        dw_c = c_ * (shrink_factor - 1)
        if dw_c + c_ != c2:
            self.cv1 = ConvBnAct(c1, c2, k, s, act=act, g=g)
            self.single_conv = True
            return

        self.cv1 = ConvBnAct(c1, c_, k, s, act=act, g=g)
        self.cv2 = ConvBnAct(c_, dw_c, dw_k, dw_s, act=act, g=c_)
        self.residual = residual

    def forward(self, x):
        if self.single_conv:
            return self.cv1(x)
        if not self.residual:
            y = self.cv1(x)
            return torch.cat((y, self.cv2(y)), 1)
        else:
            y = self.cv1(x)
            return x + torch.cat((y, self.cv2(y)), 1)


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


class RobustConv(nn.Module):
    # Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.
    def __init__(
        self,
        c1,
        c2,
        k=7,
        s=1,
        p=None,
        g=1,
        act=True,
        layer_scale_init_value=1e-6,
        residual=False,
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv, self).__init__()
        self.conv_dw = ConvBnAct(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=True)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(c2))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        y = x.to(memory_format=torch.channels_last)
        y = self.conv1x1(self.conv_dw(y))
        if self.gamma is not None:
            y = y.mul(self.gamma.reshape(1, -1, 1, 1))
        if self.residual:
            return x + y
        else:
            return x


class RobustConv2(nn.Module):
    # Robust convolution 2 (use [32, 5, 2] or [32, 7, 4] or [32, 11, 8] for one of the paths in CSP).
    def __init__(
        self, c1, c2, k=7, s=4, p=None, g=1, act=True, layer_scale_init_value=1e-6
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv2, self).__init__()
        self.conv_strided = ConvBnAct(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv_deconv = nn.ConvTranspose2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=s,
            stride=s,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(c2))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        x = self.conv_deconv(self.conv_strided(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


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

    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)

    def __repr__(self):
        s = "{name}(groups={groups})"
        return s.format(name=self.__class__.__name__, groups=self.groups)
