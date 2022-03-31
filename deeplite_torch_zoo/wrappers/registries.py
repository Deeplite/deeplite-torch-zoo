from deeplite_torch_zoo.utils.registry import DatasetWrapperRegistry
from deeplite_torch_zoo.utils.registry import ModelWrapperRegistry
from deeplite_torch_zoo.utils.registry import EvaluatorWrapperRegistry


MODEL_WRAPPER_REGISTRY = ModelWrapperRegistry()
DATA_WRAPPER_REGISTRY = DatasetWrapperRegistry()
EVAL_WRAPPER_REGISTRY = EvaluatorWrapperRegistry()
