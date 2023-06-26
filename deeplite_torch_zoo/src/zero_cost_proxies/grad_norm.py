import torch

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic


@ZERO_COST_SCORES.register('grad_norm')
def grad_norm(model, model_output_generator, loss_fn, reduction='sum'):
    model.requires_grad_(True)
    _, outputs, targets, loss_kwargs = next(model_output_generator(model))
    loss = loss_fn(outputs, targets, **loss_kwargs)
    loss.backward()

    norms = []
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norms.append(torch.norm(p.grad) ** 2)
    res = aggregate_statistic(norms, reduction=reduction)
    return res if reduction != 'sum' else torch.sqrt(norms)
