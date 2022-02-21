from deeplite_torch_zoo.utils.registry import Registry

MODEL_WRAPPER_REGISTRY = Registry('model wrapper function registry')
DATA_WRAPPER_REGISTRY = Registry('data split wrapper function registry')
EVAL_WRAPPER_REGISTRY = Registry('eval function registry')