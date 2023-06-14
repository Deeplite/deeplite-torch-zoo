import torch

from deeplite_torch_zoo.utils import get_layer_metric_array
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


@ZERO_COST_SCORES.register('plain')
def plain(model, batch, loss_fn, mode=None):
    model.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    # Gradient selection
    def plain(module):
        if module.weight.grad is not None:
            return module.weight.grad * module.weight
        else:
            return torch.zeros_like(module.weight)

    grads_abs = get_layer_metric_array(model, plain, mode)

    return sum([torch.sum(x).item() for x in grads_abs])
