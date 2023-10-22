# Code credit: https://github.com/Bobo-y/flexible-yolov5
# The file is modified by Deeplite Inc. from the original implementation on Jan 4, 2023

from functools import partial

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv, Concat
from deeplite_torch_zoo.src.dnn_blocks.yolov8.yolo_ultralytics_blocks import YOLOC3

from deeplite_torch_zoo.utils import LOGGER, make_divisible


YOLO_SCALING_GAINS = {
    'n': {'gd': 0.33, 'gw': 0.25},
    's': {'gd': 0.33, 'gw': 0.5},
    'm': {'gd': 0.67, 'gw': 0.75},
    'l': {'gd': 1, 'gw': 1},
    'x': {'gd': 1.33, 'gw': 1.25},
}


class YOLOv5FPN(nn.Module):
    """
    YOLOv5 FPN module

         concat
    C3 --->   P3
    |          ^
    V   concat | up2
    C4 --->   P4
    |          ^
    V          | up2
    C5 --->    P5
    """

    def __init__(
            self,
            ch=(256, 512, 1024),
            channel_outs=(512, 256, 256),
            version='s',
            default_gd=0.33,
            default_gw=0.5,
            bottleneck_block_cls=None,
            bottleneck_depth=3,
            act='silu',
        ):
        super().__init__()

        self.C3_size = ch[0]
        self.C4_size = ch[1]
        self.C5_size = ch[2]
        self.channels_outs = channel_outs
        self.version = version

        self.gd = default_gd
        self.gw = default_gw
        if self.version is not None and self.version.lower() in YOLO_SCALING_GAINS:
            self.gd = YOLO_SCALING_GAINS[self.version.lower()]['gd']  # depth gain
            self.gw = YOLO_SCALING_GAINS[self.version.lower()]['gw']  # width gain

        self.re_channels_out()
        self.concat = Concat()

        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5 = Conv(self.C5_size, self.channels_outs[0], 1, 1, act=act)

        if bottleneck_block_cls is None:
            bottleneck_block_cls = partial(
                YOLOC3,
                shortcut=False,
                n=self.get_depth(bottleneck_depth),
                act=act,
            )

        self.conv1 = bottleneck_block_cls(
            self.channels_outs[0] + self.C4_size,
            self.channels_outs[0],
        )

        self.P4 = Conv(self.channels_outs[0], self.channels_outs[1], 1, 1, act=act)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P3 = bottleneck_block_cls(
            self.channels_outs[1] + self.C3_size,
            self.channels_outs[1],
        )

        self.out_shape = (
            self.channels_outs[2],
            self.channels_outs[1],
            self.channels_outs[0],
        )
        LOGGER.info(
            f'FPN input channel sizes: C3 {self.C3_size}, C4 {self.C4_size}, C5 {self.C5_size}'
        )
        LOGGER.info(
            f'FPN output channel sizes: P3 {self.channels_outs[2]}, '
            f'P4 {self.channels_outs[1]}, P5 {self.channels_outs[0]}'
        )

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for idx, channel_out in enumerate(self.channels_outs):
            self.channels_outs[idx] = self.get_width(channel_out)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5 = self.P5(C5)
        up5 = self.P5_upsampled(P5)
        concat1 = self.concat([up5, C4])
        conv1 = self.conv1(concat1)

        P4 = self.P4(conv1)
        up4 = self.P4_upsampled(P4)
        concat2 = self.concat([C3, up4])

        PP3 = self.P3(concat2)

        return PP3, P4, P5
