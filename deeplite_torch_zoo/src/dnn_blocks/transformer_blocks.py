# Taken from:
# https://github.com/WongKinYiu/yolov7/blob/HEAD/models/common.py

import torch
from torch import nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct
from deeplite_torch_zoo.src.dnn_blocks.transformer_common import (
    SwinTransformerLayer, SwinTransformerLayer_v2, TransformerLayer)


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers=1, act='relu'):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = ConvBnAct(c1, c2, act=act)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers=1, window_size=8, act='relu'):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = ConvBnAct(c1, c2, act=act)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class SwinTransformer2Block(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers=1, window_size=7, act='relu'):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = ConvBnAct(c1, c2, act=act)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer_v2(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class STCSPA(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, transformer_block, n=1, act='relu', e=0.5):
        super(STCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv3 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
        num_heads = max(c_ // 32, 1)
        self.m = transformer_block(c_, c_, num_heads, n, act=act)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPB(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, transformer_block, n=1, act='relu', e=0.5):
        super(STCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c_, c_, 1, 1, act=act)
        self.cv3 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
        num_heads = max(c_ // 32, 1)
        self.m = transformer_block(c_, c_, num_heads, n, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPC(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, transformer_block, n=1, act='relu', e=0.5):
        super(STCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv3 = ConvBnAct(c_, c_, 1, 1, act=act)
        self.cv4 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
        num_heads = max(c_ // 32, 1)
        self.m = transformer_block(c_, c_, num_heads, n, act=act)

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))
