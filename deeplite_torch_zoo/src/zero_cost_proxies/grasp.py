import torch
import torch.nn as nn

from deeplite_torch_zoo.utils import get_layer_metric_array
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import compute_zc_statistic


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

    grads = get_layer_metric_array(model, grasp)

    return compute_zc_statistic(grads, reduction=reduction)
