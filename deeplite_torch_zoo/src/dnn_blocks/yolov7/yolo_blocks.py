# Taken from:
# - https://github.com/WongKinYiu/yolov7/blob/HEAD/models/common.py
# The file is modified by Deeplite Inc. from the original implementation on Feb 23, 2023
# Refactoring block implementation


import torch
import torch.nn as nn
from deeplite_torch_zoo.src.dnn_blocks.common import (
    ConvBnAct,
    DWConv,
    GhostConv,
    get_activation,
    round_channels,
)
from deeplite_torch_zoo.src.dnn_blocks.resnet.resnet_blocks import (
    GhostBottleneck,
    ResNetBottleneck,
)
from deeplite_torch_zoo.src.registries import EXPANDABLE_BLOCKS, VARIABLE_CHANNEL_BLOCKS


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOBottleneck(nn.Module):
    # Ultralytics bottleneck (2 convs)
    def __init__(
        self, c1, c2, shortcut=True, k=3, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        if g != 1:
            e = c1 / round_channels(round(c1 * e), divisor=g)
        c_ = int(c2 * e)  # hidden channels
        if c_ < g:
            return

        if isinstance(k, tuple):
            self.cv1 = ConvBnAct(c1, c_, k[0], 1, act=act)
            self.cv2 = ConvBnAct(c_, c2, k[1], 1, g=g, act=act)
        else:
            self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
            self.cv2 = ConvBnAct(c_, c2, k, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
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
        self.cv4 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = get_activation(act)
        self.m = nn.Sequential(
            *(
                YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act)
                for _ in range(n)
            )
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOBottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = get_activation(act)
        self.m = nn.Sequential(
            *(
                YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act)
                for _ in range(n)
            )
        )

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOVoVCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        act='hswish',
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = ConvBnAct(c1 // 2, c_ // 2, 3, 1, act=act)
        self.cv2 = ConvBnAct(c_ // 2, c_ // 2, 3, 1, act=act)
        self.cv3 = ConvBnAct(c_, c2, 1, 1, act=act)

    def forward(self, x):
        _, x1 = x.chunk(2, dim=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x1)
        return self.cv3(torch.cat((x1, x2), dim=1))


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOGhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=3, s=1, act='relu', shrink_factor=0.5
    ):  # ch_in, ch_out, kernel, stride
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


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
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
        self.act = get_activation(act)
        self.m = nn.Sequential(
            *[
                YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act)
                for _ in range(n)
            ]
        )

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
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
        self.act = get_activation(act)
        self.m = nn.Sequential(
            *[
                YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act)
                for _ in range(n)
            ]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.act(self.bn(torch.cat((y1, y2), dim=1)))


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOBottleneckCSPLG(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # modified by @lzrvch to have c2 out channels
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
        self.act = get_activation(act)
        self.m = nn.Sequential(
            *[
                YOLOBottleneck(g * c_, g * c_, shortcut=shortcut, g=g, e=1.0, act=act)
                for _ in range(n)
            ]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.act(self.bn(torch.cat((y1, y2), dim=1)))


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class BottleneckCSPA(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv3 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
        self.m = nn.Sequential(
            *[
                YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act)
                for _ in range(n)
            ]
        )

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class BottleneckCSPB(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c_, c_, 1, 1, act=act)
        self.cv3 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
        self.m = nn.Sequential(
            *[
                YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act)
                for _ in range(n)
            ]
        )

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class BottleneckCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv3 = ConvBnAct(c_, c_, 1, 1, act=act)
        self.cv4 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
        self.m = nn.Sequential(
            *[
                YOLOBottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0, act=act)
                for _ in range(n)
            ]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *[
                ResNetBottleneck(c_, c_, groups=g, e=0.5, act=act, shortcut=shortcut)
                for _ in range(n)
            ]
        )


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(
            *[
                ResNetBottleneck(c_, c_, groups=g, e=0.5, act=act, shortcut=shortcut)
                for _ in range(n)
            ]
        )


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *[
                ResNetBottleneck(c_, c_, groups=g, e=0.5, act=act, shortcut=shortcut)
                for _ in range(n)
            ]
        )


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class GhostCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, shrink_factor=2, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *[
                GhostBottleneck(c_, c_, shrink_factor=shrink_factor, act=act)
                for _ in range(n)
            ]
        )


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class GhostCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, shrink_factor=2, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(
            *[
                GhostBottleneck(c_, c_, shrink_factor=shrink_factor, act=act)
                for _ in range(n)
            ]
        )


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class GhostCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, shrink_factor=2, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *[
                GhostBottleneck(c_, c_, shrink_factor=shrink_factor, act=act)
                for _ in range(n)
            ]
        )


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResXCSPA(ResCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=32, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *[
                ResNetBottleneck(c_, c_, groups=g, e=1.0, act=act, shortcut=shortcut)
                for _ in range(n)
            ]
        )


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResXCSPB(ResCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=32, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(
            *[
                ResNetBottleneck(c_, c_, groups=g, e=1.0, act=act, shortcut=shortcut)
                for _ in range(n)
            ]
        )


@EXPANDABLE_BLOCKS.register()
@VARIABLE_CHANNEL_BLOCKS.register()
class ResXCSPC(ResCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=32, e=0.5, act='relu'
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, act=act)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *[
                ResNetBottleneck(c_, c_, groups=g, e=1.0, act=act, shortcut=shortcut)
                for _ in range(n)
            ]
        )


@VARIABLE_CHANNEL_BLOCKS.register()
class Stem(nn.Module):
    # Stem
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act='relu'
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Stem, self).__init__()
        c_ = int(c2 / 2)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 3, 2, act=act)
        self.cv2 = ConvBnAct(c_, c_, 1, 1, act=act)
        self.cv3 = ConvBnAct(c_, c_, 3, 2, act=act)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.cv4 = ConvBnAct(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1))


@VARIABLE_CHANNEL_BLOCKS.register()
class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2, act='relu'):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
        self.cv2 = ConvBnAct(c_, c2 // 2, 3, k, act=act)
        self.cv3 = ConvBnAct(c1, c2 // 2, 1, 1, act=act)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)
