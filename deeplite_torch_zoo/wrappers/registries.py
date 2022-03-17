from deeplite_torch_zoo.utils.registry import Registry
from deeplite_torch_zoo.utils.registry import ModelWrapperRegistry
from deeplite_torch_zoo.utils.registry import EvalWrapperRegistry



MODEL_WRAPPER_REGISTRY = ModelWrapperRegistry()
DATA_WRAPPER_REGISTRY = Registry()
EVAL_WRAPPER_REGISTRY = EvalWrapperRegistry()
