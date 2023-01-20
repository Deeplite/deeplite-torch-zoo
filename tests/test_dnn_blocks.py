import torch
import pytest
from deeplite_torch_zoo.src.dnn_blocks.ghostnet_blocks import GhostModuleV2, GhostBottleneckV2
from deeplite_torch_zoo.src.dnn_blocks.mobileone_blocks import MobileOneBlock, MobileOneBlockUnit


@pytest.mark.parametrize(
    ('block', 'c1', 'c2', 'b', 'res', 'kwargs'),
    [
        (GhostModuleV2, 64, 64, 2, 32, {"mode":"original"}),
        (GhostModuleV2, 64, 64, 2, 32, {"mode":"attn"}),
        (GhostBottleneckV2, 64, 64, 2, 32, {"use_attn":False}),
        (GhostBottleneckV2, 64, 64, 2, 32, {"use_attn":True}),
        (MobileOneBlock, 64, 64, 2, 32, {"use_se":True}),
        (MobileOneBlock, 64, 64, 2, 32, {"use_se":False}),
        (MobileOneBlockUnit, 64, 64, 2, 32, {"use_se":False}),
        (MobileOneBlockUnit, 64, 64, 2, 32, {"use_se":False}),
    ],
)
def test_blocks(block, c1, c2, b, res, kwargs):
    input = torch.rand((b, c1, res, res), device=None, requires_grad=False)
    block = block(c1, c2, **kwargs)
    output = block(input)
    
    output.sum().backward()
    print (output.shape)
    assert output.shape == (b, c2, res, res)
