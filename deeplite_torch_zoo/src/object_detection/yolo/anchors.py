from deeplite_torch_zoo.utils.registry import Registry

ANCHOR_REGISTRY = Registry()

@ANCHOR_REGISTRY.register('default')
def get_default_anchors():
    return [10, 13, 16, 30, 33, 23], \
           [30, 61, 62, 45, 59, 119], \
           [116, 90, 156, 198, 373, 326]

@ANCHOR_REGISTRY.register('default_p6')
def get_default_p6_anchors():
    return [13,17,  31,25,  24,51, 61,45], \
           [61,45,  48,102,  119,96, 97,189], \
           [97,189,  217,184,  171,384, 324,451], \
           [324,451, 545,357, 616,618, 1024,1024]
