import torch
import torch.nn as nn

from torchprofile import profile_macs

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


@ZERO_COST_SCORES.register('macs')
def macs(model, model_output_generator, loss_fn):
    inp, _, _, _ = next(model_output_generator(nn.Identity()))
    dummy_input = torch.randn(*inp.shape, device=inp.device)
    return profile_macs(model, dummy_input)
