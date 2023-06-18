import torch

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


@ZERO_COST_SCORES.register('grad_norm')
def grad_norm(model, model_output_generator, loss_fn):
    model.requires_grad_(True)
    _, outputs, targets = next(model_output_generator(model))
    loss = loss_fn(outputs, targets)
    loss.backward()

    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    return float(torch.sqrt(norm2_sum))
