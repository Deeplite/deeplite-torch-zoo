'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import numpy as np

import torch
from torch import nn

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic


def get_grad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                grad_dict[name] = [module.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                grad_dict[name].append(
                    module.weight.grad.data.cpu().reshape(-1).numpy()
                )

    return grad_dict


def compute_zico(grad_dict, mode='sum'):
    for modname in grad_dict:
        grad_dict[modname] = np.array(grad_dict[modname])

    if mode not in ('sum', 'mean'):
        raise ValueError(
            f'`mode` argument for the ZiCo metric should be one of (`sum`, `mean`), but got {mode}'
        )

    nsr_mean_abs_agg = []

    for modname in grad_dict:
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum != 0:
            if mode == 'sum':
                nsr_mean_abs_agg.append(np.log(tmpsum))
            elif mode == 'mean':
                nsr_mean_abs_agg.append(
                    np.log(np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]))
                )

    return nsr_mean_abs_agg


@ZERO_COST_SCORES.register('zico')
def zico(
    model, model_output_generator, loss_fn, n_steps=2, mode='sum', reduction='sum'
):
    grad_dict = {}
    data_generator = model_output_generator(model)

    for step in range(n_steps):
        model.zero_grad()
        _, outputs, targets, loss_kwargs = next(data_generator)
        loss = loss_fn(outputs, targets, **loss_kwargs)
        loss.backward()
        grad_dict = get_grad(model, grad_dict, step)

    return aggregate_statistic(
        compute_zico(grad_dict, mode=mode), reduction=reduction
    )
