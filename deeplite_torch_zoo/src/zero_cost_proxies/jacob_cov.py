import torch
import numpy as np

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


def get_jacob(model, model_output_generator):
    inputs, outputs, _ = next(model_output_generator(model, input_gradient=True))
    outputs.backward(torch.ones_like(outputs))
    jacob = inputs.grad.detach()
    inputs.requires_grad_(False)
    return jacob, outputs.detach()
    # return jacob, target.detach(), y.detach() # NOTE: diff with orig paper


def eval_score(jacob, k=1e-5):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    return -np.sum(np.log(v + k) + 1.0 / (v + k))


@ZERO_COST_SCORES.register('jacob_cov')
def jacob_cov(model, model_output_generator, loss_fn=None):
    model.zero_grad()
    # NOTE: diff between old/new papers
    jacobs, _ = get_jacob(model, model_output_generator)
    try:
        jc = eval_score(jacobs.reshape(jacobs.size(0), -1).cpu().numpy())
    except:
        jc = np.nan
    return jc
