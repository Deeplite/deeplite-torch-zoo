from copy import deepcopy

from deeplite_torch_zoo.src.zero_cost_proxies.fisher import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.grad_norm import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.grasp import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.jacob_cov import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.l2_norm import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.plain import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.snip import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.synflow import *  # pylint: disable=unused-import
from deeplite_torch_zoo.src.zero_cost_proxies.zico import *  # pylint: disable=unused-import

from deeplite_torch_zoo.utils import weight_gaussian_init
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


MULTI_BATCH_METRICS = ('zico',)


def get_zero_score_fn(metric_name):
    compute_score_fn = ZERO_COST_SCORES.get(metric_name)

    def compute_score_wrapper(model, dataloader, loss_fn=None, n_batches=1, **kwargs):
        model_ = deepcopy(model)
        if metric_name in MULTI_BATCH_METRICS and n_batches < 2:
            raise ValueError(
                f'Zero-cost metric {metric_name} requires n_batches > 1 but got {n_batches}'
            )
        model_.train()
        model_.zero_grad()
        if 'do_gaussian_init' in kwargs and kwargs['do_gaussian_init']:
            weight_gaussian_init(model_)
            kwargs.pop('do_gaussian_init')
        batches = [batch for idx, batch in enumerate(dataloader) if idx < n_batches]
        return compute_score_fn(model_, batches, loss_fn, **kwargs)

    return compute_score_wrapper
