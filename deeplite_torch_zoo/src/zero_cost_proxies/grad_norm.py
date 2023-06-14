import torch

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


@ZERO_COST_SCORES.register('grad_norm')
def grad_norm(model, batch, loss_fn):
    model.requires_grad_(True)
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    return float(torch.sqrt(norm2_sum))
