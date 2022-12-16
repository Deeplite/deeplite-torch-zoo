#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from typing import Optional, List

import torch
import torch.nn as nn
from deeplite_torch_zoo.src.dnn_blocks.mobileone_blocks import MobileOneBlock, reparameterize_model


__all__ = ['MobileOne', 'mobileone', 'reparameterize_model']


class MobileOne(nn.Module):
    """ MobileOne Model

        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 num_blocks_per_stage: List[int] = [2, 8, 10, 1],
                 num_classes: int = 1000,
                 width_multipliers: Optional[List[float]] = None,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(c1=3, c2=self.in_planes,
                                     k=3, s=2, p=1,
                                     inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0],
                                       num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1],
                                       num_se_blocks=0)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)

    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    num_se_blocks: int) -> nn.Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1]*(num_blocks-1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(MobileOneBlock(c1=self.in_planes,
                                         c2=self.in_planes,
                                         k=3,
                                         s=stride,
                                         p=1,
                                         g=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(MobileOneBlock(c1=self.in_planes,
                                         c2=planes,
                                         k=1,
                                         s=1,
                                         p=0,
                                         g=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    
    def get_reparameterize_model(self):
        return reparameterize_model(self)


PARAMS = {
    "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0),
           "num_conv_branches": 4},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0),
           "use_se": True},
}


def mobileone(num_classes: int = 1000, inference_mode: bool = False,
              variant: str = "s0") -> nn.Module:
    """Get MobileOne model.

    :param num_classes: Number of classes in the dataset.
    :param inference_mode: If True, instantiates model in inference mode.
    :param variant: Which type of model to generate.
    :return: MobileOne model. """
    variant_params = PARAMS[variant]
    return MobileOne(num_classes=num_classes, inference_mode=inference_mode,
                     **variant_params)
