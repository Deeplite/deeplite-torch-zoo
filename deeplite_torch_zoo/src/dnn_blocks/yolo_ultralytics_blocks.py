import functools

import numpy as np
import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ACT_TYPE_MAP, ConvBnAct
from deeplite_torch_zoo.src.dnn_blocks.yolo_blocks import (YOLOBottleneck,
                                                           YOLOGhostBottleneck)
from deeplite_torch_zoo.src.dnn_blocks.yolo_spp_blocks import YOLOSPP
from deeplite_torch_zoo.src.registries import (REPEATABLE_BLOCKS,
                                               VARIABLE_CHANNEL_BLOCKS)


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOCrossConv(nn.Module):
    # Ultralytics Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, act='relu', shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, (1, k), (1, s), act=act)
        self.cv2 = ConvBnAct(c_, c2, (k, 1), (s, 1), g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu', depth_coef=1
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv3 = ConvBnAct(2 * c_, c2, 1, act=act)  # optional act=FReLU(c2)
        n = n * depth_coef
        self.m = nn.Sequential(
            *(
                YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act)
                for _ in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = ConvBnAct(2 * self.c, c2, 1, act=act)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(YOLOBottleneck(self.c, self.c, shortcut=shortcut, g=g,
                                                k=((3, 3), (3, 3)), e=1.0, act=act) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='relu'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = ConvBnAct((2 + n) * self.c, c2, 1, act=act)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(YOLOBottleneck(self.c, self.c, shortcut=shortcut,
                                                g=g, k=((3, 3), (3, 3)), e=1.0, act=act) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC1(nn.Module):
    # CSP Bottleneck with 1 convolution
    def __init__(self, c1, c2, n=1, act='relu'):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = ConvBnAct(c1, c2, 1, 1, act=act)
        self.m = nn.Sequential(*(ConvBnAct(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        y = self.cv1(x)
        return self.m(y) + y


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3x(YOLOC3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'):
        super().__init__(c1, c2, n, shortcut, g, e, act)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(YOLOBottleneck(self.c_, self.c_, shortcut=shortcut,
                                                g=g, k=((1, 3), (3, 1)), e=1, act=act) for _ in range(n)))


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3Ghost(YOLOC3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'):
        super().__init__(c1, c2, n, shortcut, g, e, act)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(YOLOGhostBottleneck(c_, c_, act=act) for _ in range(n)))


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOCrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, act='hardswish'):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(YOLOCrossConv, self).__init__()
        Conv_ = functools.partial(ConvBnAct, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_(c1, c_, (1, k), (1, s))
        self.cv2 = Conv_(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC4(nn.Module):
    # CSP Bottleneck with 4 convolutions aka old C3
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='hardswish'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        Conv_ = functools.partial(ConvBnAct, act=act)
        CrossConv_ = functools.partial(YOLOCrossConv, act=act)
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


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3TR(YOLOC3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='hardswish'):
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        TransformerBlock_ = functools.partial(YOLOTransformerBlock, act=act)
        c_ = int(c2 * e)
        self.m = TransformerBlock_(c_, c_, 4, n)


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOC3SPP(YOLOC3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5, act='hardswish'):
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        SPP_ = functools.partial(YOLOSPP, act=act)
        c_ = int(c2 * e)
        self.m = SPP_(c_, c_, k)


@VARIABLE_CHANNEL_BLOCKS.register()
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


@REPEATABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOTransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers, act='hardswish'):
        super().__init__()
        Conv_ = functools.partial(ConvBnAct, act=act)
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


@VARIABLE_CHANNEL_BLOCKS.register()
class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True, act='leakyrelu'):
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
        self.act = ACT_TYPE_MAP[act] if act else nn.Identity()

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))
