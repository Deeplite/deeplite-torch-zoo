# This file contains modules common to various models
import math
import warnings
import functools

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU

from deeplite_torch_zoo.utils.registry import Registry
from deeplite_torch_zoo.src.objectdetection.yolov5.utils.activations import \
    Hardswish

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
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = Conv_(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


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
