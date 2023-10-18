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

import torch
import torch.nn as nn

from deeplite_torch_zoo.utils import get_layerwise_metric_values
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic


@ZERO_COST_SCORES.register('grasp')
def grasp(model, model_output_generator, loss_fn, T=1, niter=1, reduction='sum'):
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weights.append(module.weight)
            module.weight.requires_grad_(True)

    # Forward n1
    data_generator = model_output_generator(model, shuffle_data=False)
    grad_w = None
    for _ in range(niter):
        _, outputs, targets, loss_kwargs = next(data_generator)
        for i in range(len(outputs)):
            outputs[i] /= T
        loss = loss_fn(outputs, targets, **loss_kwargs)
        grad_w_p = torch.autograd.grad(loss, weights, allow_unused=True)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    # Forward n2
    _, outputs, targets, loss_kwargs = next(data_generator)
    for i in range(len(outputs)):
        outputs[i] /= T
    loss = loss_fn(outputs, targets, **loss_kwargs)
    grad_f = torch.autograd.grad(loss, weights, create_graph=True, allow_unused=True)

    # Accumulate gradients and call backwards
    z, count = 0, 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if grad_w[count] is not None:
                z += (grad_w[count].data * grad_f[count]).sum()
            count += 1
    z.backward()

    # Compute final sensitivity metric and put in gradients
    # NOTE accuracy seems to be negatively correlated with this metric (-ve)
    def grasp(module):
        if module.weight.grad is not None:
            return -module.weight.data * module.weight.grad  # -theta_q Hg
        else:
            return torch.zeros_like(module.weight)

    grads = get_layerwise_metric_values(model, grasp)

    return aggregate_statistic(grads, reduction=reduction)
