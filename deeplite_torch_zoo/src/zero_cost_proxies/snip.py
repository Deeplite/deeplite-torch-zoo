import types

import torch
import torch.nn as nn

from deeplite_torch_zoo.utils import get_layer_metric_array
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


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
def snip(model, dataloader, loss_fn, mode=None):
    inputs, targets = next(iter(dataloader))

    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight_mask = nn.Parameter(torch.ones_like(module.weight))
            module.weight.requires_grad = False
            if isinstance(module, nn.Conv2d):
                module.forward = types.MethodType(snip_forward_conv2d, module)
            else:
                module.forward = types.MethodType(snip_forward_linear, module)

    # Compute gradients, without apply
    model.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    # Gradient selection
    def snip(module):
        if module.weight_mask.grad is not None:
            return torch.abs(module.weight_mask.grad)
        else:
            return torch.zeros_like(module.weight)

    grads_abs = get_layer_metric_array(model, snip, mode)

    return sum([torch.sum(x).item() for x in grads_abs])
