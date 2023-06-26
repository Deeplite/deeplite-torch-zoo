import torch

from deeplite_torch_zoo.utils import get_layer_metric_array
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic


@ZERO_COST_SCORES.register('l2_norm')
def get_l2_norm_array(
    model, model_output_generator=None, loss_fn=None, reduction='sum'
):
    norm = get_layer_metric_array(model, lambda l: l.weight.norm())
    return aggregate_statistic(norm, reduction=reduction)
