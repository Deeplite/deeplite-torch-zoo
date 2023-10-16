# main zoo API
from deeplite_torch_zoo.api import (  # pylint: disable=unused-import
    get_model,
    get_dataloaders,
    get_eval_function,
    list_models_by_dataset,
    list_models,
    profile,
    create_model,  # deprecated
    get_model_by_name,  # deprecated
    get_data_splits_by_name,  # deprecated
)
from deeplite_torch_zoo.api.eval.zero_cost import get_zero_cost_estimator

# utils
from deeplite_torch_zoo.utils import LOGGER  # pylint: disable=unused-import

from . import postinstall

postinstall.initialize()
