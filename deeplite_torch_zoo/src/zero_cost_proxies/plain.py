import torch

from deeplite_torch_zoo.utils import get_layer_metric_array
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import compute_zc_statistic


@ZERO_COST_SCORES.register('plain')
def plain(model, model_output_generator, loss_fn, reduction='sum'):
    model.zero_grad()

    _, outputs, targets, loss_kwargs = next(model_output_generator(model))
    loss = loss_fn(outputs, targets, **loss_kwargs)
    loss.backward()

    # gradient selection
    def plain(module):
        if module.weight.grad is not None:
            return module.weight.grad * module.weight
        else:
            return torch.zeros_like(module.weight)

    grads_abs = get_layer_metric_array(model, plain)
    return compute_zc_statistic(grads_abs, reduction=reduction)
