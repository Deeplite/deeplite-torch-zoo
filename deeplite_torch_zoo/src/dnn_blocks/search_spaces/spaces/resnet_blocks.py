from deeplite_torch_zoo.src.dnn_blocks.resnet.resnet_blocks import (
    ResNetBasicBlock,
    ResNetBottleneck,
    ResNeXtBottleneck,
)
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.shufflenet_blocks import ShuffleUnit
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.squeezenet_blocks import FireUnit, SqnxtUnit

from deeplite_torch_zoo.src.dnn_blocks.search_spaces.block_registry import (
    DNNBlockRegistry,
)

SE_RATIO_RANGE = (None, 8, 16, 32)
EXPANSION_FACTOR_RANGE = (0.05, 0.1, 0.15, 0.2, 0.25)

SMALL_BLOCK_REGISTRY = DNNBlockRegistry('small')


SMALL_BLOCK_REGISTRY.register(
    name='ResNetBottleneck',
    e=(0.05, 0.1, 0.15, 0.2, 0.25),
    se_ratio=(None,),
)(ResNetBottleneck)


RESNET_BLOCK_REGISTRY = DNNBlockRegistry('resnet')


RESNET_BLOCK_REGISTRY.register(
    name='ResNetBottleneck',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    se_ratio=SE_RATIO_RANGE,
)(ResNetBottleneck)


RESNET_BLOCK_REGISTRY.register(
    name='ResNeXtBottleneck',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    se_ratio=SE_RATIO_RANGE,
    groups=(16, 32),
)(ResNeXtBottleneck, flags=('groupconv', ))


RESNET_BLOCK_REGISTRY.register(
    name='ResNetBasicBlock',
    k=(1, 3),
    se_ratio=SE_RATIO_RANGE,
)(ResNetBasicBlock)


RESNET_BLOCK_REGISTRY.register(
    name='ShuffleNetBlock',
    dw_k=(3, 5),
    e=(2, 4, 8),
    g=(1, 2, 4),
)(ShuffleUnit, flags=('groupconv', ))


RESNET_BLOCK_REGISTRY.register(
    name='SqueezeNetBlock',
    k=(3, 5),
    e=(0.5, 0.25, 0.125),
)(FireUnit)


RESNET_BLOCK_REGISTRY.register(
    name='SqueezeNextBlock',
    k=(3, 5),
    e=(1, 2, 4),
)(SqnxtUnit)
