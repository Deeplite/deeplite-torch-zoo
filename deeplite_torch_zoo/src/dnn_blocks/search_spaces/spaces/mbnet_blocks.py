from deeplite_torch_zoo.src.dnn_blocks.mbnet.mbconv_blocks import MBConv
from deeplite_torch_zoo.src.dnn_blocks.effnet.effnet_blocks import FusedMBConv
from deeplite_torch_zoo.src.dnn_blocks.search_spaces.block_registry import (
    DNNBlockRegistry,
)
from deeplite_torch_zoo.src.dnn_blocks.search_spaces.spaces import INCLUDE_DW


MBNET_BLOCK_REGISTRY = DNNBlockRegistry('mobilenet')

if INCLUDE_DW:
    MBNET_BLOCK_REGISTRY.register(
        name='MBConv',
        e=(1, 4, 6),
        k=(3, 5),
        se_ratio=(None, 8, 16, 32),
    )(MBConv)


    MBNET_BLOCK_REGISTRY.register(
        name='FusedMBConv',
        e=(1, 4, 6),
        k=(1, 3, 5),
        se_ratio=(None, 8, 16, 32),
    )(FusedMBConv)
