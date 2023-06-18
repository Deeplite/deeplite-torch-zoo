from copy import deepcopy
from itertools import repeat

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


def get_zero_score_estimator(metric_name):
    compute_score_fn = ZERO_COST_SCORES.get(metric_name)

    def compute_score_wrapper(model, loss_fn=None, model_output_generator=None, dataloader=None, **kwargs):
        model_ = deepcopy(model)
        
        if model_output_generator is None:
            def default_model_output_generator(model, shuffle_data=True):
                loader = dataloader if shuffle_data else repeat(next(iter(dataloader)))
                for inputs, targets in loader:
                    outputs = model(inputs)  
                    yield inputs, outputs, targets

            model_output_generator = default_model_output_generator

        model_.train()
        model_.zero_grad()
        
        if 'do_gaussian_init' in kwargs and kwargs['do_gaussian_init']:
            weight_gaussian_init(model_)
            kwargs.pop('do_gaussian_init')
        
        return compute_score_fn(model_, default_model_output_generator, loss_fn, **kwargs)

    return compute_score_wrapper
