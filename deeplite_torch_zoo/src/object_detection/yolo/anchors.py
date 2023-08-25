from deeplite_torch_zoo.utils.registry import Registry

ANCHOR_REGISTRY = Registry()

@ANCHOR_REGISTRY.register('default')
def get_default_anchors():
    return [10, 13, 16, 30, 33, 23], \
           [30, 61, 62, 45, 59, 119], \
           [116, 90, 156, 198, 373, 326]
