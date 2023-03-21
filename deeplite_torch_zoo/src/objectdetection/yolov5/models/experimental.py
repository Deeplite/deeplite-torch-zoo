# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# This file contains experimental modules

import functools
import math

import numpy as np
import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv
from deeplite_torch_zoo.src.dnn_blocks.common import DWConv
from deeplite_torch_zoo.src.dnn_blocks.yolov7.yolo_blocks import YOLOSPP as SPP
from deeplite_torch_zoo.src.dnn_blocks.yolov7.yolo_blocks import \
    YOLOBottleneck as Bottleneck


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, act='hardswish'):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        Conv_ = functools.partial(Conv, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, (1, k), (1, s))
        self.cv2 = Conv_(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_old(nn.Module):
    # CSP Bottleneck with 4 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='hardswish'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        Conv_ = functools.partial(Conv, act=act)
        CrossConv_ = functools.partial(CrossConv, act=act)
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


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='hardswish'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        Conv_ = functools.partial(Conv, act=act)
        Bottleneck_ = functools.partial(Bottleneck, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, 1, 1)
        self.cv2 = Conv_(c1, c_, 1, 1)
        self.cv3 = Conv_(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_(c1=c_, c2=c_, shortcut=shortcut, g=g, e=1.0) for _ in range(n)))

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


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='hardswish'):
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        TransformerBlock_ = functools.partial(TransformerBlock, act=act)
        c_ = int(c2 * e)
        self.m = TransformerBlock_(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5, act='hardswish'):
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        SPP_ = functools.partial(SPP, act=act)
        c_ = int(c2 * e)
        self.m = SPP_(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='hardswish'):
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        GhostBottleneck_ = functools.partial(GhostBottleneck, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck_(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1, act='hardswish'):  # ch_in, ch_out, kernel, stride
        super().__init__()
        Conv_ = functools.partial(Conv, act=act)
        GhostConv_ = functools.partial(GhostConv, act=act)
        DWConv_ = functools.partial(DWConv, act=act)
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv_(c1, c_, 1, 1),  # pw
                                  DWConv_(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv_(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv_(c1, c1, k, s, act=False),
                                      Conv_(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act='hardswish'):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        Conv_ = functools.partial(Conv, act=act)
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


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers, act='hardswish'):
        super().__init__()
        Conv_ = functools.partial(Conv, act=act)
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


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import (
        Detect, YOLOModel)

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
        ckpt = torch.load(w, map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, YOLOModel):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model
