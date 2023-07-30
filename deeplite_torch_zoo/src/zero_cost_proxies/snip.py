# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import types

import torch
import torch.nn as nn

from deeplite_torch_zoo.utils import get_layerwise_metric_values
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic


def snip_forward_conv2d(self, x):
    return nn.functional.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_linear(self, x):
    return nn.functional.linear(x, self.weight * self.weight_mask, self.bias)


@ZERO_COST_SCORES.register('snip')
def snip(model, model_output_generator, loss_fn, reduction='sum'):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight_mask = nn.Parameter(torch.ones_like(module.weight))
            module.weight.requires_grad = False
            if isinstance(module, nn.Conv2d):
                module.forward = types.MethodType(snip_forward_conv2d, module)
            else:
                module.forward = types.MethodType(snip_forward_linear, module)

    # Compute gradients, without apply
    _, outputs, targets, loss_kwargs = next(model_output_generator(model))
    loss = loss_fn(outputs, targets, **loss_kwargs)
    loss.backward()

    # Gradient selection
    def snip(module):
        if module.weight_mask.grad is not None:
            return torch.abs(module.weight_mask.grad)
        else:
            return torch.zeros_like(module.weight)

    grads_abs = get_layerwise_metric_values(model, snip,
                                            target_layer_types=(nn.Conv2d, nn.Linear))
    return aggregate_statistic(grads_abs, reduction=reduction)
