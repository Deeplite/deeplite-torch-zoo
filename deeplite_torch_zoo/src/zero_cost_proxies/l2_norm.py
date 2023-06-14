from deeplite_torch_zoo.utils import get_layer_metric_array
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


@ZERO_COST_SCORES.register('l2_norm')
def get_l2_norm_array(model, batch, loss_fn=None, mode='param', eval_mode=False):
    return get_layer_metric_array(
        model if not eval_mode else model.eval(), lambda l: l.weight.norm(), mode=mode
    )
