# This file contains experimental modules

import functools

import numpy as np
import torch
import torch.nn as nn

from deeplite_torch_zoo.src.objectdetection.yolov5.models.common import \
    Conv, DWConv, Bottleneck, SPP, CUSTOM_ACTIVATION_MODULES
from deeplite_torch_zoo.src.objectdetection.yolov5.utils.google_utils import \
    attempt_download


@CUSTOM_ACTIVATION_MODULES.register('CrossConv')
class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, activation_type='hardswish'):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, (1, k), (1, s))
        self.cv2 = Conv_(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@CUSTOM_ACTIVATION_MODULES.register('C3')
class C3(nn.Module):
    # CSP Bottleneck with 4 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, activation_type='hardswish'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        CrossConv_ = functools.partial(CrossConv, activation_type=activation_type)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv_(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(
            *[CrossConv_(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@CUSTOM_ACTIVATION_MODULES.register('C3v6')
class C3v6(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, activation_type='hardswish'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        Bottleneck_ = functools.partial(Bottleneck, activation_type=activation_type)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = Conv_(c1, c_, 1, 1)
        self.cv3 = Conv_(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(
                -torch.arange(1.0, n) / 2, requires_grad=True
            )  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


@CUSTOM_ACTIVATION_MODULES.register('C3TR')
class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, activation_type='hardswish'):
        super().__init__(c1, c2, n, shortcut, g, e, activation_type=activation_type)
        TransformerBlock_ = functools.partial(TransformerBlock, activation_type=activation_type)
        c_ = int(c2 * e)
        self.m = TransformerBlock_(c_, c_, 4, n)


@CUSTOM_ACTIVATION_MODULES.register('C3SPP')
class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5, activation_type='hardswish'):
        super().__init__(c1, c2, n, shortcut, g, e, activation_type=activation_type)
        SPP_ = functools.partial(SPP, activation_type=activation_type)
        c_ = int(c2 * e)
        self.m = SPP_(c_, c_, k)


@CUSTOM_ACTIVATION_MODULES.register('C3Ghost')
class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, activation_type='hardswish'):
        super().__init__(c1, c2, n, shortcut, g, e, activation_type=activation_type)
        GhostBottleneck_ = functools.partial(GhostBottleneck, activation_type=activation_type)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck_(c_, c_) for _ in range(n)))


@CUSTOM_ACTIVATION_MODULES.register('GhostBottleneck')
class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1, activation_type='hardswish'):  # ch_in, ch_out, kernel, stride
        super().__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        GhostConv_ = functools.partial(GhostConv, activation_type=activation_type)
        DWConv_ = functools.partial(DWConv, activation_type=activation_type)
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv_(c1, c_, 1, 1),  # pw
                                  DWConv_(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv_(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv_(c1, c1, k, s, act=False),
                                      Conv_(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


@CUSTOM_ACTIVATION_MODULES.register('GhostConv')
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, activation_type='hardswish'):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv_(c1, c_, k, s, None, g, act)
        self.cv2 = Conv_(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


@CUSTOM_ACTIVATION_MODULES.register('TransformerBlock')
class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers, activation_type='hardswish'):
        super().__init__()
        Conv_ = functools.partial(Conv, activation_type=activation_type)
        self.conv = None
        if c1 != c2:
            self.conv = Conv_(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[
                0
            ].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [
                nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False)
                for g in range(groups)
            ]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(
            torch.load(w, map_location=map_location)["model"].float().fuse().eval()
        )  # load FP32 model

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
