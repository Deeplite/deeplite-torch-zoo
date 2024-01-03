from deeplite_torch_zoo.utils.registry import RegistryStorage

from deeplite_torch_zoo.src.dnn_blocks.search_spaces.block_registry import (
    DNNBlockRegistry,
)
from deeplite_torch_zoo.src.dnn_blocks.search_spaces.spaces.generic_blocks import (
    GENERIC_BLOCK_REGISTRY,
)
from deeplite_torch_zoo.src.dnn_blocks.search_spaces.spaces.mbnet_blocks import (
    MBNET_BLOCK_REGISTRY,
)
from deeplite_torch_zoo.src.dnn_blocks.search_spaces.spaces.resnet_blocks import (
    RESNET_BLOCK_REGISTRY,
    SMALL_BLOCK_REGISTRY,
)
from deeplite_torch_zoo.src.dnn_blocks.search_spaces.spaces.yolo_blocks import (
    YOLO_BLOCK_REGISTRY,
)


CNN_BLOCKS_REGISTRY = DNNBlockRegistry('cnn')
CNN_BLOCKS_REGISTRY += (
    GENERIC_BLOCK_REGISTRY
    + RESNET_BLOCK_REGISTRY
    + MBNET_BLOCK_REGISTRY
    + YOLO_BLOCK_REGISTRY
)

FULL_BLOCKS_REGISTRY = DNNBlockRegistry('full')
FULL_BLOCKS_REGISTRY += CNN_BLOCKS_REGISTRY

SEARCH_SPACES = RegistryStorage(
    [
        SMALL_BLOCK_REGISTRY,
        GENERIC_BLOCK_REGISTRY,
        RESNET_BLOCK_REGISTRY,
        MBNET_BLOCK_REGISTRY,
        YOLO_BLOCK_REGISTRY,
        CNN_BLOCKS_REGISTRY,
        FULL_BLOCKS_REGISTRY,
    ]
)
