from torch import nn

from deeplite_torch_zoo.src.dnn_blocks.search_spaces.spaces import INCLUDE_DW
from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct, DWConv
from deeplite_torch_zoo.src.dnn_blocks.ghostnetv2.ghostnet_blocks import GhostConv
from deeplite_torch_zoo.src.dnn_blocks.search_spaces.block_registry import (
    DNNBlockRegistry,
)

GENERIC_BLOCK_REGISTRY = DNNBlockRegistry('generic')

GENERIC_BLOCK_REGISTRY.register(name='Identity')(nn.Identity)

GENERIC_BLOCK_REGISTRY.register(
    name='Conv',
    k=(1, 3, 5),
)(ConvBnAct)


if INCLUDE_DW:
    GENERIC_BLOCK_REGISTRY.register(
        name='DWConv',
        k=(3, 5, 7),
    )(DWConv)


    GENERIC_BLOCK_REGISTRY.register(
        name='GhostConv',
        k=(1, 3),
        dw_k=(3, 5),
        dfc=(True, False),
    )(GhostConv)
