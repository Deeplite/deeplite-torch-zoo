import torch
import numpy as np

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


def get_jacob(model, x):
    x.requires_grad_(True)
    y = model(x)
    y = torch.cat([yi.view(y[0].shape[0], 3, -1) for yi in y], 2)  # Double check
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    x.requires_grad_(False)
    return jacob, y.detach()
    # return jacob, target.detach(), y.detach() # NOTE: diff with orig paper


def eval_score(jacob, k=1e-5):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    return -np.sum(np.log(v + k) + 1.0 / (v + k))


@ZERO_COST_SCORES.register('jacob_cov')
def jacob_cov(model, batch, loss_fn=None):
    model.zero_grad()
    # NOTE: diff between old/new papers
    inputs, _ = batch
    jacobs, _ = get_jacob(model, inputs)
    try:
        jc = eval_score(jacobs.reshape(jacobs.size(0), -1).cpu().numpy())
    except:
        jc = np.nan
    return jc
