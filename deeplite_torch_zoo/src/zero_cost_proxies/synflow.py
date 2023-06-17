import torch

from deeplite_torch_zoo.utils import get_layer_metric_array
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


@ZERO_COST_SCORES.register('synflow')
def synflow(model, dataloader, loss_fn=None, mode=None):
    # Convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # Convert to orig values
    @torch.no_grad()
    def nonlinearize(model, signs):
        for name, param in model.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])
        return

    model.zero_grad()
    signs = linearize(model)  # Keep signs of all params
    model.zero_grad()
    model.double()

    inputs, _ = next(iter(dataloader))

    # Compute gradients with input of all-ones
    shape = list(inputs.shape[1:])
    device = inputs.device
    inputs = torch.ones([1] + shape, device=device, dtype=torch.float64)
    outputs = model.forward(inputs)
    torch.sum(torch.cat([x.flatten() for x in outputs])).backward()

    # Select the gradients
    def get_synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs_list = get_layer_metric_array(model, get_synflow, mode)
    nonlinearize(model, signs)  # Apply signs of all params
    score = 0
    for grad_abs in grads_abs_list:
        if len(grad_abs.shape) == 4:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1, 2, 3])))
        elif len(grad_abs.shape) == 2:
            score += float(torch.mean(torch.sum(grad_abs, dim=[1])))

    return score
