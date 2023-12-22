import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import (
    ConvBnAct,
    get_activation,
    round_channels,
)
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.cnn_attention import SELayer


class FusedMBConv(nn.Module):
    # Taken from https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    def __init__(
        self, c1, c2, e=1.0, k=3, stride=1, act='relu', se_ratio=None, channel_divisor=1, shortcut=True,
    ):
        super().__init__()
        assert stride in (1, 2)
        self.shortcut = shortcut
        c_ = round_channels(c1 * e, channel_divisor)
        self.conv = nn.Sequential(
            # pw
            ConvBnAct(c1, c_, k, stride, act=None),
            # Squeeze-and-Excite
            SELayer(c_, reduction=se_ratio) if se_ratio else nn.Identity(),
            # activation
            get_activation(act),
            # pw-linear
            ConvBnAct(c_, c2, 1, 1, act=None),
        )

    def forward(self, x):
        return self.conv(x) if not self.shortcut else x + self.conv(x)
