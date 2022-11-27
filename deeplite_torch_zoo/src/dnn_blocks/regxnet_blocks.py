from functools import partial

import torch
import torch.nn as nn
from deeplite_torch_zoo.src.dnn_blocks.common import (ACT_TYPE_MAP, ConvBnAct,
                                                      SELayer, _make_divisible)

SEWithNorm = partial(SELayer, norm_layer=nn.BatchNorm2d)


class RexNetBottleneck(nn.Module):
    def __init__(self, c1, c2, stride=1, exp_ratio=1.0, k=3, se_ratio=None, ch_div=1,
            act='hswish', dw_act='relu6'):
        super(RexNetBottleneck, self).__init__()

        self.in_channels = c1
        self.out_channels = c2

        if exp_ratio != 1.:
            dw_chs = _make_divisible(round(c1 * exp_ratio), divisor=ch_div)
            self.conv_exp = ConvBnAct(c1, dw_chs, act=act)
        else:
            dw_chs = c1
            self.conv_exp = None

        self.conv_dw = ConvBnAct(dw_chs, dw_chs, k, s=stride, g=dw_chs, act=None)

        self.se = SEWithNorm(dw_chs, rd_channels=_make_divisible(int(dw_chs * se_ratio), ch_div)) if se_ratio is not None \
            else nn.Identity()

        self.se = SELayer(dw_chs, reduction=se_ratio) if se_ratio else nn.Identity()

        self.act_dw = ACT_TYPE_MAP[dw_act] if dw_act else nn.Identity()

        self.conv_pwl = ConvBnAct(dw_chs, c2, 1, act=None)

    def forward(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        x = torch.cat([x[:, 0:self.in_channels] + shortcut,
            x[:, self.in_channels:]], dim=1)
        return x
