import pytest

import torch

from deeplite_torch_zoo import get_model_by_name
from deeplite_torch_zoo.src.dnn_blocks.search_spaces import SEARCH_SPACES


block_registry = SEARCH_SPACES.get('full')

BLOCK_VALIDITY_TESTS = []
for block_key in list(block_registry.registry_dict.keys()):
    BLOCK_VALIDITY_TESTS.append(
        (
            'resnet12',
            (2, 3, 224, 224),
            block_key
        )
    )


@pytest.mark.parametrize(
    ('model_name', 'input_shape', 'block_key'),
    BLOCK_VALIDITY_TESTS
)
def test_block_validity(model_name, input_shape, block_key):
    model = get_model_by_name(
        model_name=model_name,
        dataset_name='imagenet',
        device='cpu'
    )
    block_cls = block_registry.get(block_key)
    model.features.stage1.unit2 = block_cls(64, 64)
    y = model(torch.randn(*input_shape))
    y.sum().backward()
