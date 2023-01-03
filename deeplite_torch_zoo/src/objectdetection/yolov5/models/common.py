# This file contains modules common to various models
import functools
import math
import warnings

import torch
import torch.nn as nn
from deeplite_torch_zoo.src.objectdetection.yolov5.utils.activations import \
    Hardswish
from deeplite_torch_zoo.utils.registry import Registry
from torch.nn.modules.activation import LeakyReLU

from deeplite_torch_zoo.src.dnn_blocks.common import *
from deeplite_torch_zoo.src.dnn_blocks.yolo_blocks import *

try:
    from mish_cuda import MishCuda as Mish
except:

    class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
        def forward(self, x):
            return x * torch.nn.functional.softplus(x).tanh()


# Registry for the modules that contain custom activation fn inside
CUSTOM_ACTIVATION_MODULES = Registry()

ACTIVATION_FN_NAME_MAP = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'hardswish': Hardswish,
    'mish': Mish,
    'leakyrelu': nn.LeakyReLU,
    'leakyrelu_0.1': functools.partial(nn.LeakyReLU, negative_slope=0.1),
}


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True, activation_type='hardswish'):
    # Depthwise convolution
    Conv_ = functools.partial(Conv, activation_type=activation_type)
    return Conv_(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


@CUSTOM_ACTIVATION_MODULES.register('Conv')
class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True, activation_type='hardswish'
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if activation_type not in ACTIVATION_FN_NAME_MAP:
            raise ValueError(f'Activation type {activation_type} not available.'
                             f'Available activation types: {list(ACTIVATION_FN_NAME_MAP.keys())}')
        activation_function = ACTIVATION_FN_NAME_MAP[activation_type]
        self.act = activation_function() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

@CUSTOM_ACTIVATION_MODULES.register('Bottleneck')
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, g=1, e=0.5, activation_type='hardswish',
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()

        if activation_type=='hardswish':
            activation_type = 'hswish'
        # Conv_ = functools.partial(ConvBnAct, act=activation_type)
        # c_ = int(c2 * e)  # hidden channels
        # self.cv1 = Conv_(c1, c_, 1, 1)
        # self.cv2 = Conv_(c_, c2, 3, 1, g=g)
        # self.add = shortcut and c1 == c2
        self.Conv_ = YOLOBottleneck(c1=c1, c2=c2, shortcut=shortcut, g=g, e=e, act=activation_type)

    def forward(self, x):
        return self.Conv_(x) #x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@CUSTOM_ACTIVATION_MODULES.register('BottleneckCSP')
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, activation_type='hardswish',
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        Bottleneck_ = functools.partial(Bottleneck, activation_type=activation_type)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv_(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(
            *[Bottleneck_(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@CUSTOM_ACTIVATION_MODULES.register('BottleneckCSP2')
class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, activation_type='hardswish',
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        Bottleneck_ = functools.partial(Bottleneck, activation_type=activation_type)
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv_(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.m = nn.Sequential(
            *[Bottleneck_(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@CUSTOM_ACTIVATION_MODULES.register('BottleneckCSP2Leaky')
class BottleneckCSP2Leaky(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, activation_type='hardswish',
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2Leaky, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        Bottleneck_ = functools.partial(Bottleneck, activation_type=activation_type)
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv_(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = LeakyReLU(negative_slope=0.1)
        self.m = nn.Sequential(
            *[Bottleneck_(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@CUSTOM_ACTIVATION_MODULES.register('VoVCSP')
class VoVCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, activation_type='hardswish',
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(VoVCSP, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv_(c1 // 2, c_ // 2, 3, 1)
        self.cv2 = Conv_(c_ // 2, c_ // 2, 3, 1)
        self.cv3 = Conv_(c_, c2, 1, 1)

    def forward(self, x):
        _, x1 = x.chunk(2, dim=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x1)
        return self.cv3(torch.cat((x1, x2), dim=1))


@CUSTOM_ACTIVATION_MODULES.register('SPP')
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), activation_type='hardswish'):
        super(SPP, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = Conv_(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


@CUSTOM_ACTIVATION_MODULES.register('SPPCSP')
class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), activation_type='hardswish'):
        super(SPPCSP, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv_(c_, c_, 3, 1)
        self.cv4 = Conv_(c_, c_, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = Conv_(4 * c_, c_, 1, 1)
        self.cv6 = Conv_(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.cv7 = Conv_(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@CUSTOM_ACTIVATION_MODULES.register('SPPCSPLeaky')
class SPPCSPLeaky(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), activation_type='hardswish'):
        super(SPPCSPLeaky, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv_(c_, c_, 3, 1)
        self.cv4 = Conv_(c_, c_, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = Conv_(4 * c_, c_, 1, 1)
        self.cv6 = Conv_(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = LeakyReLU(negative_slope=0.1)
        self.cv7 = Conv_(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@CUSTOM_ACTIVATION_MODULES.register('Focus')
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True, activation_type='hardswish',
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        self.conv = Conv_(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
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


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p), groups=g, bias=False
        )  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat(
            [self.aap(y) for y in (x if isinstance(x, list) else [x])], 1
        )  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


@CUSTOM_ACTIVATION_MODULES.register('SPPF')
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, activation_type='hardswish'):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = Conv_(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


@CUSTOM_ACTIVATION_MODULES.register('SPPCSPC')
class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), activation_type='hardswish'):
        super(SPPCSPC, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = Conv_(c1, c_, 1, 1)
        self.cv3 = Conv_(c_, c_, 3, 1)
        self.cv4 = Conv_(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv_(4 * c_, c_, 1, 1)
        self.cv6 = Conv_(c_, c_, 3, 1)
        self.cv7 = Conv_(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


@CUSTOM_ACTIVATION_MODULES.register('RepConv')
class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, activation_type='hardswish', deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        if activation_type not in ACTIVATION_FN_NAME_MAP:
            raise ValueError(f'Activation type {activation_type} not available.'
                             f'Available activation types: {list(ACTIVATION_FN_NAME_MAP.keys())}')
        activation_function = ACTIVATION_FN_NAME_MAP[activation_type]
        self.act = activation_function() if act else nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups,
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )


        #print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        #print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        #print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None
