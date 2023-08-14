# Credit: https://github.com/BayesWatch/nas-without-training

from functools import partial

import torch
import numpy as np

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.dnn_blocks.common import ACT_TYPE_MAP
from deeplite_torch_zoo.utils import TRAINABLE_LAYERS


ACTIVATION_TYPES = tuple(type(module) for module in ACT_TYPE_MAP.values())


def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return 0 if (np.isneginf(ld) and s == 0) else ld


@ZERO_COST_SCORES.register('nwot')
def compute_nwot(
    model,
    model_output_generator,
    loss_fn=None,
    reduction='sum',
    pre_act=False,
    ):
    model.eval()
    inputs, _, _, _ = next(model_output_generator(model))

    def counting_forward_hook(module, inp, out):
        if pre_act:
            out = out.view(out.size(0), -1)
            x = (out > 0).float()
        else:
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()

        K = x @ x.t()
        K2 = (1. - x) @ (1. - x.t())
        if reduction == 'sum':
            model.K += K.cpu().numpy() + K2.cpu().numpy()  # hamming distance
        elif reduction == None:
            model.K.append(logdet(K.cpu().numpy() + K2.cpu().numpy()))

    model.K = 0. if reduction == 'sum' else []
    for module in model.modules():
        if (pre_act and isinstance(module, TRAINABLE_LAYERS)) or (not pre_act and isinstance(module, ACTIVATION_TYPES)):
            module.register_forward_hook(counting_forward_hook)

    with torch.no_grad():
        model(inputs)

    return logdet(model.K) if reduction == 'sum' else model.K


ZERO_COST_SCORES.register('nwot_preact')(partial(compute_nwot, pre_act=True))
