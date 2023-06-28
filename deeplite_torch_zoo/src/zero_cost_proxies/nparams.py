from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic


@ZERO_COST_SCORES.register('nparams')
def nparams(
    model, model_output_generator, loss_fn, include_frozen_params=False, reduction='sum'
):
    nparams_array = [
        p.numel()
        for p in model.parameters()
        if include_frozen_params or p.requires_grad
    ]
    return aggregate_statistic(nparams_array, reduction=reduction)
