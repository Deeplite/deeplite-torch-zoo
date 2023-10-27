# Code credit: https://github.com/Bobo-y/flexible-yolov5
# The file is modified by Deeplite Inc. from the original implementation on Jan 4, 2023

from functools import partial

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import Concat
from deeplite_torch_zoo.src.dnn_blocks.yolov8.yolo_ultralytics_blocks import YOLOC2f
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.neck_utils import YOLO_SCALING_GAINS

from deeplite_torch_zoo.utils import LOGGER, make_divisible


class YOLOv8FPN(nn.Module):
    """
    YOLOv8 FPN module

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
            channel_outs=(512, 256),
            version='s',
            default_gd=0.33,
            default_gw=0.5,
            bottleneck_block_cls=None,
            bottleneck_depth=3,
            act='silu',
            channel_divisor=8,
        ):
        super().__init__()

        self.C3_size = ch[0]
        self.C4_size = ch[1]
        self.C5_size = ch[2]
        self.channels_outs = channel_outs
        self.channel_divisor = channel_divisor
        self.version = version

        self.gd = default_gd
        self.gw = default_gw
        if self.version is not None and self.version.lower() in YOLO_SCALING_GAINS:
            self.gd = YOLO_SCALING_GAINS[self.version.lower()]['gd']  # depth gain
            self.gw = YOLO_SCALING_GAINS[self.version.lower()]['gw']  # width gain

        self.re_channels_out()
        self.concat = Concat()

        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        if bottleneck_block_cls is None:
            bottleneck_block_cls = partial(
                YOLOC2f,
                shortcut=False,
                n=self.get_depth(bottleneck_depth),
                act=act,
            )

        self.conv1 = bottleneck_block_cls(
            self.C5_size + self.C4_size,
            self.channels_outs[0]
        )

        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.P3 = bottleneck_block_cls(
            self.channels_outs[0] + self.C3_size,
            self.channels_outs[1],
        )

        self.out_shape = (
            self.channels_outs[1],
            self.channels_outs[0],
            self.C5_size
        )
        LOGGER.info(
            f'FPN input channel sizes: C3 {self.C3_size}, C4 {self.C4_size}, C5 {self.C5_size}'
        )
        LOGGER.info(
            f'FPN output channel sizes: P3 {self.channels_outs[1]}, '
            f'P4 {self.channels_outs[0]}, P5 {self.C5_size}'
        )

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, self.channel_divisor)

    def re_channels_out(self):
        for idx, channel_out in enumerate(self.channels_outs):
            self.channels_outs[idx] = self.get_width(channel_out)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        up5 = self.P5_upsampled(C5)
        concat1 = self.concat([up5, C4])
        P4 = self.conv1(concat1)

        up4 = self.P4_upsampled(P4)
        concat2 = self.concat([C3, up4])

        PP3 = self.P3(concat2)

        return PP3, P4, C5
