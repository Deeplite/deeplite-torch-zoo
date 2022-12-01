import torch
import torch.nn as nn

ACT_TYPE_MAP = {
    'relu': nn.ReLU(inplace=True),
    'relu6': nn.ReLU6(inplace=True),
    'hswish': nn.Hardswish(),
    'silu': nn.SiLU(inplace=True),
    'lrelu': nn.LeakyReLU(0.1, inplace=True),
    'sigmoid': nn.Sigmoid(),
}


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def _make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class ConvBnAct(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act='relu', d=1, residual=False, use_bn=True,
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.in_channels = c1
        self.out_channels = c2

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d, groups=g, bias=False)

        self.bn = nn.BatchNorm2d(c2) if use_bn else nn.Identity()
        self.act = ACT_TYPE_MAP[act] if act else nn.Identity()
        self.residual = residual

    def forward(self, x):
        if not self.residual:
            return self.act(self.bn(self.conv(x)))
        else:
            return x + self.act(self.bn(self.conv(x)))


class DWConv(ConvBnAct):
    # Depth-wise convolution class
    def __init__(
        self, c1, c2, k, s=1, act='relu', residual=False,
    ):  # ch_in, kernel, stride, padding, groups
        if c1 != c2:
            raise ValueError('Input and output channel count of DWConv does not match')
        super().__init__(c1, c2, k, s, g=c1, act=act, residual=residual)


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act='relu', dw_k=3, dw_s=1,
        residual=False, shrink_factor=2,
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


class SELayerDeprecated(nn.Module):
    # Deprecated
    def __init__(self, channel, reduction=4):
        super(SELayerDeprecated, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                nn.Hardswish()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SELayer(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, reduction=16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            bias=True, act='relu', norm_layer=None, gate_layer='sigmoid'):
        super(SELayer, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = _make_divisible(channels / reduction, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = ACT_TYPE_MAP[act] if act else nn.Identity()
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = ACT_TYPE_MAP[gate_layer] if gate_layer else nn.Identity()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act='relu'):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = ConvBnAct(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2],
            x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))


class RobustConv(nn.Module):
    # Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.
    def __init__(self, c1, c2, k=7, s=1, p=None, g=1,
        act=True, layer_scale_init_value=1e-6, residual=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv, self).__init__()
        self.conv_dw = ConvBnAct(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) \
            if layer_scale_init_value > 0 else None

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
    def __init__(self, c1, c2, k=7, s=4, p=None, g=1,
        act=True, layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv2, self).__init__()
        self.conv_strided = ConvBnAct(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv_deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=s, stride=s,
                                              padding=0, bias=True, dilation=1, groups=1
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = self.conv_deconv(self.conv_strided(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
