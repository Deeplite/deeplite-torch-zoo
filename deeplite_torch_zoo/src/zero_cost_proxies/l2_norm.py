import torch

from deeplite_torch_zoo.utils import get_layer_metric_array
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


@ZERO_COST_SCORES.register('l2_norm')
def get_l2_norm_array(model, dataloader, loss_fn=None, mode='param', eval_mode=False):
    norm = get_layer_metric_array(
        model if not eval_mode else model.eval(),
        lambda l: l.weight.norm(), mode=mode
    )
    return sum([torch.sum(x).item() for x in norm])
