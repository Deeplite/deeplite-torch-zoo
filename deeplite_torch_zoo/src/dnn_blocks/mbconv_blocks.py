import torch.nn as nn
from deeplite_torch_zoo.src.dnn_blocks.common import (ACT_TYPE_MAP, ConvBnAct,
                                                      DWConv, SELayer)


class MBConv(nn.Module):
    # https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py
    def __init__(self, c1, c2, e=1.0, k=3, stride=1, act='relu', se_ratio=None):
        super().__init__()
        assert stride in [1, 2]

        if e == 1.0:
            self.conv = nn.Sequential(
                # dw
                DWConv(c1, c1, k, stride, act=None),
                # Squeeze-and-Excite
                SELayer(c1, reduction=se_ratio) if se_ratio else nn.Identity(),
                # activation
                ACT_TYPE_MAP[act] if act else nn.Identity(),
                # pw-linear
                ConvBnAct(c1, c2, 1, 1, act=None),
            )
        else:
            c_ = round(c1 * e)
            self.conv = nn.Sequential(
                # pw
                ConvBnAct(c1, c_, 1, 1, act=act),
                # dw
                DWConv(c_, c_, k, stride, act=None),
                # Squeeze-and-Excite
                SELayer(c_, reduction=se_ratio) if se_ratio else nn.Identity(),
                # activation
                ACT_TYPE_MAP[act] if act else nn.Identity(),
                # pw-linear
                ConvBnAct(c_, c2, 1, 1, act=None),
            )

    def forward(self, x):
        return x + self.conv(x)


class FusedMBConv(nn.Module):
    # https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    def __init__(self, c1, c2, e=1.0, k=3, stride=1, act='relu', se_ratio=None):
        super().__init__()
        assert stride in [1, 2]

        c_ = round(c1 * e)
        self.conv = nn.Sequential(
            # pw
            ConvBnAct(c1, c_, k, stride, act=None),
            # Squeeze-and-Excite
            SELayer(c_, reduction=se_ratio) if se_ratio else nn.Identity(),
            # activation
            ACT_TYPE_MAP[act] if act else nn.Identity(),
            # pw-linear
            ConvBnAct(c_, c2, 1, 1, act=None),
        )

    def forward(self, x):
        return x + self.conv(x)
