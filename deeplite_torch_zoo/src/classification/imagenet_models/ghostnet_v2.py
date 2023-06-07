import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplite_torch_zoo.src.dnn_blocks.common import round_channels, ConvBnAct
from deeplite_torch_zoo.src.dnn_blocks.ghostnetv2.ghostnet_blocks import (
    GhostBottleneckV2,
)


class GhostNetV2(nn.Module):
    def __init__(
        self,
        cfgs,
        num_classes=1000,
        width=1.0,
        dropout=0.2,
        block=GhostBottleneckV2,
        args=None,
    ):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = round_channels(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        # block = block
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = round_channels(c * width, 4)
                hidden_channel = round_channels(exp_size * width, 4)
                if block == GhostBottleneckV2:
                    layers.append(
                        block(
                            input_channel,
                            hidden_channel,
                            output_channel,
                            k,
                            s,
                            se_ratio=se_ratio,
                            layer_id=layer_id,
                            args=args,
                        )
                    )
                input_channel = output_channel
                layer_id += 1
            stages.append(nn.Sequential(*layers))

        output_channel = round_channels(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet_v2(num_classes=1000, width=1.0, **kwargs):
    cfgs = [
        # k, t, c, SE, s
        [[3, 16, 16, 0, 1]],
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        [[3, 240, 80, 0, 2]],
        [
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1],
        ],
        [[5, 672, 160, 0.25, 2]],
        [
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
        ],
    ]
    return GhostNetV2(
        cfgs,
        num_classes=num_classes,
        width=width,
        dropout=kwargs['dropout'],
        args=kwargs['args'],
    )
