# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import numpy as np

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic


def get_jacob(model, model_output_generator, output_post_processing):
    inputs, outputs, _, _ = next(model_output_generator(model, input_gradient=True))
    outputs = output_post_processing(outputs)
    outputs.backward(torch.ones_like(outputs))
    jacob = inputs.grad.detach()
    inputs.requires_grad_(False)
    return jacob, outputs.detach()
    # return jacob, target.detach(), y.detach() # NOTE: diff with orig paper


def eval_score(jacob, k=1e-5):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    return -(np.log(v + k) + 1.0 / (v + k))


@ZERO_COST_SCORES.register('jacob_cov')
def jacob_cov(
    model,
    model_output_generator,
    loss_fn=None,
    output_post_processing=None,
    reduction='sum',
):
    if output_post_processing is None:
        output_post_processing = lambda tensors: torch.cat(
            [x.flatten() for x in tensors]
        )
    # NOTE: diff between old/new papers
    jacobs, _ = get_jacob(model, model_output_generator, output_post_processing)
    try:
        jc = eval_score(jacobs.reshape(jacobs.size(0), -1).cpu().numpy())
    except:
        jc = np.nan
    return aggregate_statistic(jc, reduction=reduction)
