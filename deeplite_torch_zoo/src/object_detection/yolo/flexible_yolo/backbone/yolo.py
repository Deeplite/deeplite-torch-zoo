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
    ):
        super().__init__()

        self.version = version
        self.gd = YOLO_SCALING_GAINS[self.version.lower()]['gd']  # depth gain
        self.gw = YOLO_SCALING_GAINS[self.version.lower()]['gw']  # width gain

        if bottleneck_block_cls is None:
            bottleneck_block_cls = partial(
                YOLOC3,
                act=act,
            )

        self.channels_out = [64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024]
        self.re_channels_out()

        self.C1 = Conv(3, self.channels_out[0], 6, 2, 2)

        self.C2 = Conv(self.channels_out[0], self.channels_out[1], 3, 2)
        self.conv1 = bottleneck_block_cls(
            self.channels_out[1], self.channels_out[2], self.get_depth(num_blocks[0])
        )

        self.C3 = Conv(self.channels_out[2], self.channels_out[3], 3, 2)
        self.conv2 = bottleneck_block_cls(
            self.channels_out[3], self.channels_out[4], self.get_depth(num_blocks[1])
        )

        self.C4 = Conv(self.channels_out[4], self.channels_out[5], 3, 2)
        self.conv3 = bottleneck_block_cls(
            self.channels_out[5], self.channels_out[6], self.get_depth(num_blocks[2])
        )

        self.C5 = Conv(self.channels_out[6], self.channels_out[7], 3, 2)

        self.conv4 = bottleneck_block_cls(
            self.channels_out[7], self.channels_out[8], self.get_depth(num_blocks[3])
        )
        self.sppf = YOLOSPPF(self.channels_out[8], self.channels_out[9], 5)

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
    ):
        if bottleneck_block_cls is None:
            bottleneck_block_cls = partial(
                YOLOC2f,
                shortcut=True,
                act=act,
            )
        super().__init__(
            version=version,
            num_blocks=num_blocks,
            bottleneck_block_cls=bottleneck_block_cls,
            act=act,
        )
        self.C1 = Conv(3, self.channels_out[0], 3, 2)
