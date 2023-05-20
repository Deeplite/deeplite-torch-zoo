import warnings

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct, get_activation
from deeplite_torch_zoo.src.registries import VARIABLE_CHANNEL_BLOCKS


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), act='hswish'):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c_ * (len(k) + 1), c2, 1, 1, act=act)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), act='hswish'
    ):
        super().__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = ConvBnAct(c_, c_, 3, 1, act=act)
        self.cv4 = ConvBnAct(c_, c_, 1, 1, act=act)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = ConvBnAct(4 * c_, c_, 1, 1, act=act)
        self.cv6 = ConvBnAct(c_, c_, 3, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = get_activation(act)
        self.cv7 = ConvBnAct(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPPCSPLeaky(nn.Module):
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), act='leakyrelu'
    ):
        super().__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = ConvBnAct(c_, c_, 3, 1, act=act)
        self.cv4 = ConvBnAct(c_, c_, 1, 1, act=act)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = ConvBnAct(4 * c_, c_, 1, 1, act=act)
        self.cv6 = ConvBnAct(c_, c_, 3, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = get_activation(act)
        self.cv7 = ConvBnAct(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, act='hswish'):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c_ * 4, c2, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), act='hswish'
    ):
        super().__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv3 = ConvBnAct(c_, c_, 3, 1, act=act)
        self.cv4 = ConvBnAct(c_, c_, 1, 1, act=act)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = ConvBnAct(4 * c_, c_, 1, 1, act=act)
        self.cv6 = ConvBnAct(c_, c_, 3, 1, act=act)
        self.cv7 = ConvBnAct(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
