
import torch
import torch.nn as nn
from deeplite_torch_zoo.src.dnn_blocks.common import (ACT_TYPE_MAP, ConvBnAct,
                                                      DWConv, GhostConv,
                                                      _make_divisible)
from deeplite_torch_zoo.src.dnn_blocks.resnet_blocks import (GhostBottleneck,
                                                             ResXNetBottleneck)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, c1, c2, kernel_sizes=(5, 9, 13), act="relu"
    ):
        super().__init__()
        c_ = c1 // 2
        self.conv1 = ConvBnAct(c1, c_, k=1, s=1, act=act)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        self.conv2 = ConvBnAct(c_*4, c2, k=1, s=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class YOLOBottleneck(nn.Module):
    # Ultralytics bottleneck (2 convs)
    def __init__(
        self, c1, c2, k=3, shortcut=True, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        if g != 1:
            e = c1 / _make_divisible(round(c1 * e), divisor=g)
        c_ = int(c2 * e)  # hidden channels
        if c_ < g:
            return
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c_, c2, k, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


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


class YOLOBottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = ConvBnAct(2 * c_, c2, 1, 1,  act=act)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = ACT_TYPE_MAP[act] if act else nn.Identity()
        self.m = nn.Sequential(
            *(YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act) for _ in range(n))
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


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
            *(YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class YOLOGhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1, act='relu', shrink_factor=0.5):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1, act=act, shrink_factor=shrink_factor),  # pw
            DWConv(c_, c_, k, s, act='') if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act='', shrink_factor=shrink_factor),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=''), ConvBnAct(c1, c2, 1, 1, act=''))
            if s == 1
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class YOLOBottleneckCSPF(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        # self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = ACT_TYPE_MAP[act] if act else nn.Identity()
        self.m = nn.Sequential(
            *[YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class YOLOBottleneckCSPL(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # modified by @ivan-lazarevich to have c2 out channels
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c1, c2 - c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        # self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(c2)  # applied to cat(cv2, cv3)
        self.act = ACT_TYPE_MAP[act] if act else nn.Identity()
        self.m = nn.Sequential(
            *[YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.act(self.bn(torch.cat((y1, y2), dim=1)))


class YOLOBottleneckCSPLG(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # modified by @ivan-lazarevich to have c2 out channels
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=3, e=0.25, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, g * c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c1, c2 - g * c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(g * c_, g * c_, 1, 1, groups=g, bias=False)
        # self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(c2)  # applied to cat(cv2, cv3)
        self.act = ACT_TYPE_MAP[act] if act else nn.Identity()
        self.m = nn.Sequential(
            *[YOLOBottleneck(g * c_, g * c_, shortcut=shortcut, g=g, e=1.0, act=act) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.act(self.bn(torch.cat((y1, y2), dim=1)))


class BottleneckCSPA(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1)
        self.cv2 = ConvBnAct(c1, c_, 1, 1)
        self.cv3 = ConvBnAct(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPB(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1)
        self.cv2 = ConvBnAct(c_, c_, 1, 1)
        self.cv3 = ConvBnAct(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1)
        self.cv2 = ConvBnAct(c1, c_, 1, 1)
        self.cv3 = ConvBnAct(c_, c_, 1, 1)
        self.cv4 = ConvBnAct(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class ResCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[ResXNetBottleneck(c_, c_, groups=g, e=0.5) for _ in range(n)])


class ResCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[ResXNetBottleneck(c_, c_, groups=g, e=0.5) for _ in range(n)])


class ResCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[ResXNetBottleneck(c_, c_, groups=g, e=0.5) for _ in range(n)])


class GhostCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, shrink_factor=2):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_, shrink_factor=shrink_factor) for _ in range(n)])


class GhostCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, shrink_factor=2):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_, shrink_factor=shrink_factor) for _ in range(n)])


class GhostCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, shrink_factor=2):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_, shrink_factor=shrink_factor) for _ in range(n)])
