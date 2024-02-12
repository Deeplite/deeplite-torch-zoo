import inspect
from functools import partial

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.search_spaces.block_registry import (
    DNNBlockRegistry,
)
from deeplite_torch_zoo.src.dnn_blocks.search_spaces.spaces.yolo_blocks import (
    YOLO_BLOCK_REGISTRY,
)

META_BLOCK_REGISTRY = DNNBlockRegistry('metablocks')


class MetaBottleneck(nn.Module):
    def __init__(self, c1, c2, block_cls1=None, block_cls2=None, e=0.5, act='silu', shortcut=True):
        super().__init__()
        c_ = int(c1 * e)
        kw1 = {}
        kw2 = {}
        if 'shortcut' in inspect.signature(block_cls1).parameters:
            kw1 = {'shortcut': shortcut}
        if 'shortcut' in inspect.signature(block_cls2).parameters:
            kw2 = {'shortcut': shortcut}

        self.block1 = block_cls1(c1, c_, act=act, **kw1)
        self.block2 = block_cls2(c_, c2, act=act, **kw2)

    def forward(self, x):
        return self.block2(self.block1(x))


block_types = []
block_registry = YOLO_BLOCK_REGISTRY
block_keys = list(block_registry.registry_dict.keys())
for block_key in block_keys:
    block_cls = block_registry.get(block_key)
    block_types.append(block_cls)

block_cls_pairs = [(block_cls1, block_cls2) for block_cls1 in block_keys for block_cls2 in block_keys
                   if block_cls1 == block_cls2]

for block_cls_pair in block_cls_pairs:
    block_cls1 = block_registry.get(block_cls_pair[0])
    block_cls2 = block_registry.get(block_cls_pair[1])

    META_BLOCK_REGISTRY.register(
        name=f'MetaBottleneck_{str(block_cls1)}_{str(block_cls2)}',
        e=(0.5, 0.75, 1.0),
    )(partial(MetaBottleneck, block_cls1=block_cls1, block_cls2=block_cls2))
