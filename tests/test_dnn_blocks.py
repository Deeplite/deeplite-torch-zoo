import torch
import pytest
from deeplite_torch_zoo.src.dnn_blocks.ghostnet_blocks import GhostModuleV2, GhostBottleneckV2
from deeplite_torch_zoo.src.dnn_blocks.mobileone_blocks import MobileOneBlock, MobileOneBlockUnit
from deeplite_torch_zoo.src.dnn_blocks.squeezenet_blocks import FireUnit, SqnxtUnit
from deeplite_torch_zoo.src.dnn_blocks.repvgg_blocks import RepConv
from deeplite_torch_zoo.src.dnn_blocks.mbconv_blocks import MBConv, FusedMBConv
from deeplite_torch_zoo.src.dnn_blocks.pelee_blocks import TwoStackDenseBlock
from deeplite_torch_zoo.src.dnn_blocks.large_kernel_blocks import RepLKBlock
from deeplite_torch_zoo.src.dnn_blocks.regxnet_blocks import RexNetBottleneck


@pytest.mark.parametrize(
    ('block', 'c1', 'c2', 'b', 'res', 'kwargs'),
    [
        (GhostModuleV2, 64, 64, 2, 32, {"mode":"original"}),
        (GhostModuleV2, 64, 64, 2, 32, {"mode":"attn"}),

        (GhostBottleneckV2, 64, 64, 2, 32, {"use_attn":False}),
        (GhostBottleneckV2, 64, 64, 2, 32, {"use_attn":True}),

        (MobileOneBlock, 64, 64, 2, 32, {"use_se":True}),
        (MobileOneBlock, 64, 64, 2, 32, {"use_se":False}),

        (MobileOneBlockUnit, 64, 64, 2, 32, {"use_se":True}),
        (MobileOneBlockUnit, 64, 64, 2, 32, {"use_se":False}),

        (FireUnit, 64, 64, 2, 32, {}),
        (SqnxtUnit, 64, 64, 2, 32, {}),

        (RepConv, 64, 64, 2, 32, {}),
        (MBConv, 64, 64, 2, 32, {}),
        (FusedMBConv, 64, 64, 2, 32, {}),

        (TwoStackDenseBlock, 64, 64, 2, 32, {}),

        (RepLKBlock, 64, 64, 2, 32, {}),
        (RexNetBottleneck, 64, 64, 2, 32, {}),
    ],
)


def test_blocks(block, c1, c2, b, res, kwargs):
    input = torch.rand((b, c1, res, res), device=None, requires_grad=True)
    block = block(c1, c2, **kwargs)
    output = block(input)
    
    output.sum().backward()
    print (output.shape)
    assert output.shape == (b, c2, res, res)
