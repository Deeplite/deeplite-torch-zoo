import types

import torch
import torch.nn as nn

from deeplite_torch_zoo.utils import get_layer_metric_array, reshape_elements
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


def fisher_forward_conv2d(self, x):
    # Get activations after passing through 'hooked' identity op
    x = nn.functional.conv2d(
        x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
    )
    self.act = self.dummy(x)
    return self.act


def fisher_forward_linear(self, x):
    x = nn.functional.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act


@ZERO_COST_SCORES.register('fisher')
def fisher(model, dataloader, loss_fn, mode='channel'):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Variables/op needed for fisher computation
            layer.fisher = None
            layer.act = 0.0
            layer.dummy = nn.Identity()

            # Replace forward ops conv/linear
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(fisher_forward_linear, layer)

            # Backward hook on identity op at layer output
            def hook_factory(layer):
                def hook(module, grad_input, grad_output):
                    act = layer.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        g_nk = torch.sum((act * grad), list(range(2, len(act.shape))))
                    else:
                        g_nk = act * grad
                    del_k = g_nk.pow(2).mean(0).mul(0.5)
                    if layer.fisher is None:
                        layer.fisher = del_k
                    else:
                        layer.fisher += del_k
                    del layer.act

                return hook

            layer.dummy.register_backward_hook(hook_factory(layer))

    inputs, targets = next(iter(dataloader))
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    # Retrieve fisher info
    def fisher(module):
        if module.fisher is not None:
            return torch.abs(module.fisher.detach())
        else:
            return torch.zeros(module.weight.shape[0])  # size=ch

    grads_abs_ch = get_layer_metric_array(model, fisher, mode)
    shapes = get_layer_metric_array(model, lambda l: l.weight.shape[1:], mode)
    grads_abs = reshape_elements(grads_abs_ch, shapes, inputs.device)

    return sum([torch.sum(x).item() for x in grads_abs])
