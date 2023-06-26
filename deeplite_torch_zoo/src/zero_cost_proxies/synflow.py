import torch
import torch.nn as nn

from deeplite_torch_zoo.utils import get_layer_metric_array, NORMALIZATION_LAYERS
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic


@ZERO_COST_SCORES.register('synflow')
def synflow(
    model,
    model_output_generator,
    loss_fn=None,
    dummify_bns=False,
    bn_training_mode=True,
    reduction='sum',
    output_post_processing=None,
):
    if output_post_processing is None:
        output_post_processing = lambda tensors: torch.cat(
            [x.flatten() for x in tensors]
        )

    # replace *norm layer forwards with dummy forwards
    if dummify_bns:

        def dummify_bns_fn(module):
            if isinstance(module, NORMALIZATION_LAYERS):
                module.forward = lambda x: x

        model.apply(dummify_bns_fn)

    # convert params to their abs
    @torch.no_grad()
    def linearize(model):
        for param in model.state_dict().values():
            param.abs_()

    model.double()
    model.zero_grad()

    if not bn_training_mode:
        if not dummify_bns:
            model.eval()
        linearize(model)

    inp, _, _, _ = next(model_output_generator(nn.Identity()))
    # compute gradients with input of all-ones
    inputs = torch.ones((1, *inp.shape[1:]), device=inp.device, dtype=torch.float64)
    outputs = model(inputs)
    torch.sum(output_post_processing(outputs)).backward()

    # select the gradients
    def get_synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(model, get_synflow)
    return aggregate_statistic(grads_abs, reduction=reduction)
