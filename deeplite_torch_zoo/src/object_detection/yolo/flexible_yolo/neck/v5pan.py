# Code credit: https://github.com/Bobo-y/flexible-yolov5

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


class YOLOv5PAN(nn.Module):
    """
    YOLOv5 PAN module
    P3 --->  PP3
    ^         |
    | concat  V
    P4 --->  PP4
    ^         |
    | concat  V
    P5 --->  PP5
    """

    def __init__(
        self,
        ch=(256, 256, 512),
        channel_outs=(256, 512, 512, 1024),
        version='s',
        default_gd=0.33,
        default_gw=0.5,
        bottleneck_block_cls=None,
        bottleneck_depth=3,
        act='silu',
    ):
        super().__init__()
        self.version = str(version)
        self.channels_outs = channel_outs

        self.gd = default_gd
        self.gw = default_gw
        if self.version is not None and self.version.lower() in YOLO_SCALING_GAINS:
            self.gd = YOLO_SCALING_GAINS[self.version.lower()]['gd']  # depth gain
            self.gw = YOLO_SCALING_GAINS[self.version.lower()]['gw']  # width gain

        self.re_channels_out()

        self.P3_size = ch[0]
        self.P4_size = ch[1]
        self.P5_size = ch[2]

        if bottleneck_block_cls is None:
            bottleneck_block_cls = partial(
                YOLOC3,
                shortcut=False,
                n=self.get_depth(bottleneck_depth),
                act=act,
            )

        first_stride = 2
        self.convP3 = Conv(self.P3_size, self.channels_outs[0], 3, first_stride, act=act)
        self.P4 = bottleneck_block_cls(
            self.channels_outs[0] + self.P4_size,
            self.channels_outs[1],
        )

        second_stride = 2
        self.convP4 = Conv(self.channels_outs[1], self.channels_outs[2], 3, second_stride, act=act)
        self.P5 = bottleneck_block_cls(
            self.channels_outs[2] + self.P5_size,
            self.channels_outs[3],
        )

        self.concat = Concat()
        self.out_shape = [self.P3_size, self.channels_outs[2], self.channels_outs[3]]
        LOGGER.info(
            'PAN input channel size: P3 {}, P4 {}, P5 {}'.format(
                self.P3_size, self.P4_size, self.P5_size
            )
        )
        LOGGER.info(
            'PAN output channel size: PP3 {}, PP4 {}, PP5 {}'.format(
                self.P3_size, self.channels_outs[2], self.channels_outs[3]
            )
        )

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for idx, channel_out in enumerate(self.channels_outs):
            self.channels_outs[idx] = self.get_width(channel_out)

    def forward(self, inputs):
        PP3, P4, P5 = inputs

        convp3 = self.convP3(PP3)
        concat3_4 = self.concat([convp3, P4])
        PP4 = self.P4(concat3_4)

        convp4 = self.convP4(PP4)
        concat4_5 = self.concat([convp4, P5])
        PP5 = self.P5(concat4_5)

        return PP3, PP4, PP5
