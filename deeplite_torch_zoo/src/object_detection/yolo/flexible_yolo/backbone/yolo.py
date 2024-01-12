from functools import partial

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv
from deeplite_torch_zoo.src.dnn_blocks.yolov8.yolo_ultralytics_blocks import (
    YOLOC3,
    YOLOC2f,
)
from deeplite_torch_zoo.src.dnn_blocks.yolov7.yolo_spp_blocks import YOLOSPPF
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.neck_utils import (
    YOLO_SCALING_GAINS,
)

from deeplite_torch_zoo.utils import LOGGER, make_divisible


class YOLOv5Backbone(nn.Module):
    def __init__(
        self,
        version='s',
        num_blocks=(3, 6, 9, 3),
        bottleneck_block_cls=None,
        act='silu',
        channels_out=(64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024),
        depth_factor=None,
        width_factor=None,
    ):
        super().__init__()

        Conv_ = partial(Conv, act=act)

        self.version = version
        if depth_factor is None:
            self.gd = YOLO_SCALING_GAINS[self.version.lower()]['gd']  # depth gain
        else:
            self.gd = depth_factor
        if width_factor is None:
            self.gw = YOLO_SCALING_GAINS[self.version.lower()]['gw']  # width gain
        else:
            self.gw = width_factor

        if bottleneck_block_cls is None:
            bottleneck_block_cls = [
                partial(
                    YOLOC3,
                    act=act,
                    n=self.get_depth(n),
                )
                for n in num_blocks
            ]

        self.channels_out = list(channels_out)
        self.re_channels_out()

        self.C1 = Conv_(3, self.channels_out[0], 6, 2, 2)

        self.C2 = Conv_(self.channels_out[0], self.channels_out[1], 3, 2)
        self.conv1 = bottleneck_block_cls[0](
            self.channels_out[1], self.channels_out[2]
        )

        self.C3 = Conv_(self.channels_out[2], self.channels_out[3], 3, 2)
        self.conv2 = bottleneck_block_cls[1](
            self.channels_out[3], self.channels_out[4]
        )

        self.C4 = Conv_(self.channels_out[4], self.channels_out[5], 3, 2)
        self.conv3 = bottleneck_block_cls[2](
            self.channels_out[5], self.channels_out[6]
        )

        self.C5 = Conv_(self.channels_out[6], self.channels_out[7], 3, 2)

        self.conv4 = bottleneck_block_cls[3](
            self.channels_out[7], self.channels_out[8]
        )
        self.sppf = YOLOSPPF(
            self.channels_out[8],
            self.channels_out[9],
            k=5,
            act=act,
        )

        self.out_shape = [
            self.channels_out[3],
            self.channels_out[5],
            self.channels_out[9],
        ]

        LOGGER.info(
            f'Backbone output channel: C3 {self.channels_out[3]},'
            f'C4 {self.channels_out[5]}, C5(SPPF) {self.channels_out[9]}'
        )

    def forward(self, x):
        c1 = self.C1(x)

        c2 = self.C2(c1)
        conv1 = self.conv1(c2)

        c3 = self.C3(conv1)
        conv2 = self.conv2(c3)

        c4 = self.C4(conv2)
        conv3 = self.conv3(c4)

        c5 = self.C5(conv3)
        conv4 = self.conv4(c5)

        sppf = self.sppf(conv4)

        return c3, c4, sppf

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for idx, channel_out in enumerate(self.channels_out):
            self.channels_out[idx] = self.get_width(channel_out)


class YOLOv8Backbone(YOLOv5Backbone):
    def __init__(
        self,
        version='s',
        num_blocks=(3, 6, 6, 3),
        bottleneck_block_cls=None,
        act='silu',
        channels_out=(64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024),
        depth_factor=None,
        width_factor=None,
    ):
        self.version = version
        Conv_ = partial(Conv, act=act)
        if depth_factor is None:
            self.gd = YOLO_SCALING_GAINS[self.version.lower()]['gd']  # depth gain
        else:
            self.gd = depth_factor
        if bottleneck_block_cls is None:
            bottleneck_block_cls = [
                partial(
                    YOLOC2f,
                    shortcut=True,
                    act=act,
                    n=self.get_depth(n),
                )
                for n in num_blocks
            ]
        super().__init__(
            version=version,
            num_blocks=num_blocks,
            bottleneck_block_cls=bottleneck_block_cls,
            act=act,
            channels_out=channels_out,
            depth_factor=depth_factor,
            width_factor=width_factor,
        )
        self.C1 = Conv_(3, self.channels_out[0], 3, 2)
