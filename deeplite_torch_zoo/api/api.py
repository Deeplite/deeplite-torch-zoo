import collections
import fnmatch

import texttable
import torch
from torchinfo import summary
from torchprofile import profile_macs

import deeplite_torch_zoo.api.datasets  # pylint: disable=unused-import
import deeplite_torch_zoo.api.eval  # pylint: disable=unused-import
import deeplite_torch_zoo.api.models  # pylint: disable=unused-import
from deeplite_torch_zoo.utils import switch_train_mode, LOGGER
from deeplite_torch_zoo.api.registries import (
    DATA_WRAPPER_REGISTRY,
    EVAL_WRAPPER_REGISTRY,
    MODEL_WRAPPER_REGISTRY,
)

__all__ = [
    "get_data_splits_by_name",
    "get_model_by_name",
    "get_eval_function",
    "list_models",
    "create_model",
    "profile",
    "get_models_by_dataset",
]


def get_data_splits_by_name(data_root, dataset_name, model_name, **kwargs):
    """
    The datasets function calls in the format of (get_`dataset_name`_for_`model_name`).
    Except for classification since the datasets format for classification models is the same.
    The function calls for classification models are in the format (get_`dataset_name`)

    returns datasplits in the following format:
    {
       'train': train_data_loader,
       'test' : test_data_loader
    }
    """
    data_split_wrapper_fn = DATA_WRAPPER_REGISTRY.get(
        dataset_name=dataset_name, model_name=model_name
    )
    data_split = data_split_wrapper_fn(data_root=data_root, **kwargs)
    return data_split


def get_model_by_name(
    model_name,
    dataset_name,
    pretrained=True,
    fp16=False,
    device="cuda",
    **kwargs,
):
    """
    Tries to find a matching model creation wrapper function in the registry and uses it to create a new model object
    :param model_name: Name of the model to create
    :param dataset_name: Name of dataset the model was trained / is to be trained on
    :param pretrained: Whether to load pretrained weights
    :param fp16: Whether to convert the model to fp16 precision
    :param device: Loads the model either on a gpu (`cuda`, `cuda:device_id`) or cpu.

    returns a corresponding model object (optionally with pretrained weights)
    """
    model_func = MODEL_WRAPPER_REGISTRY.get(
        model_name=model_name.lower(), dataset_name=dataset_name
    )
    model = model_func(
        pretrained=pretrained, device=device, **kwargs
    )
    return model.half() if fp16 else model


def get_eval_function(model_name, dataset_name):
    task_type = MODEL_WRAPPER_REGISTRY.get_task_type(
        model_name=model_name, dataset_name=dataset_name
    )
    eval_function = EVAL_WRAPPER_REGISTRY.get(
        task_type=task_type, model_name=model_name, dataset_name=dataset_name
    )
    return eval_function


def create_model(
    model_name,
    pretraining_dataset,
    num_classes=None,
    pretrained=False,
    fp16=False,
    device="cuda",
    **kwargs,
):
    """
    Tries to find a matching model creation wrapper function in the registry (for the corresponding model name
    and pretraining dataset name) and uses it to create a new model object, optionally with a custom number
    of output classes

    :param model_name: Name of the model to create
    :param pretraining_dataset: Name of pretraining dataset to (partially) load the weights from
    :param num_classes: Number of output classes in the new model
    :param fp16: Whether to convert the model to fp16 precision
    :param device: Loads the model either on a gpu (`cuda`, `cuda:device_id`) or cpu.

    returns a corresponding model object (optionally with a custom number of classes)
    """
    model_func = MODEL_WRAPPER_REGISTRY.get(
        model_name=model_name.lower(), dataset_name=pretraining_dataset
    )
    model_wrapper_kwargs = {
        'pretrained': pretrained,
        'device': device,
        **kwargs,
    }
    if num_classes is not None:
        model_wrapper_kwargs.update({'num_classes': num_classes})
    model = model_func(**model_wrapper_kwargs)
    return model.half() if fp16 else model


def profile(model, img_size=224, in_ch=3, verbose=False, fuse=True):
    """
    Do model profiling to calculate the MAC count, the number of parameters and weight size of the model.
    The torchprofile package is used to calculate MACs, which is jit-based and
    is more accurate than the hook based approach.

    :param model: PyTorch nn.Module object
    :param img_size: Input image resolution, either an integer value or a tuple of integers of the form (3, 224, 224)
    :param in_ch: Number of input channels to use in case passed img_size is an integer
    :param verbose: if True, prints the model summary table from torchinfo

    returns a dictionary with the number of GMACs, number of parameters in millions and model size in megabytes
    """
    device = next(model.parameters()).device

    if not isinstance(img_size, tuple):
        img_size = (in_ch, img_size, img_size)

    if fuse and hasattr(model, 'fuse'):
        model = model.fuse()

    with switch_train_mode(model, is_training=False):
        model_stats = summary(model, input_size=(1, *img_size), verbose=verbose)
        model_size_mb = model_stats.to_megabytes(model_stats.total_param_bytes)
        model_mparams = model_stats.total_params / 1e6

        macs = profile_macs(model.to(device), torch.randn(1, *img_size).to(device))
        model_gmacs = macs / 1e9

    model.to(device)
    return {'GMACs': model_gmacs, 'size_Mb': model_size_mb, 'Mparams': model_mparams}


def list_models(
    filter='',
    print_table=True,
    return_list=False,
    task_type_filter=None,
    include_no_checkpoint=False,
):
    """
    A helper function to list all existing models or dataset calls
    It takes a `model_name` or a `dataset_name` as a filter and
    prints a table of corresponding available models

    :param filter: a string or list of strings containing model name, dataset name or "model_name_dataset_name"
    to use as a filter
    :param print_table: Whether to print a table with matched models to the console
    :param return_list: Whether to return a list with model names and corresponding datasets
    """
    filter = '*' + filter + '*'
    if include_no_checkpoint:
        all_model_keys = MODEL_WRAPPER_REGISTRY.registry_dict.keys()
    else:
        all_model_keys = MODEL_WRAPPER_REGISTRY.pretrained_models.keys()
    if task_type_filter is not None:
        allowed_task_types = set(MODEL_WRAPPER_REGISTRY.task_type_map.values())
        if task_type_filter not in allowed_task_types:
            raise RuntimeError(
                f'Wrong task type filter value. Allowed values are {allowed_task_types}'
            )
        all_model_keys = [
            model_key
            for model_key in all_model_keys
            if MODEL_WRAPPER_REGISTRY.task_type_map[model_key] == task_type_filter
        ]
    all_models = {
        model_key.model_name + '_' + model_key.dataset_name: model_key
        for model_key in all_model_keys
    }
    models = []
    include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
    for f in include_filters:
        include_models = fnmatch.filter(all_models.keys(), f)
        if include_models:
            models = set(models).union(include_models)
    found_model_keys = [all_models[model] for model in sorted(models)]

    if print_table:
        table = texttable.Texttable()
        rows = collections.defaultdict(list)
        for model_key in found_model_keys:
            rows[model_key.model_name].extend([model_key.dataset_name])
        for model in rows:
            rows[model] = ', '.join(rows[model])
        table.add_rows([['Available models', 'Source datasets'], *rows.items()])
        LOGGER.info(table.draw())

    return found_model_keys if return_list else None


def get_models_by_dataset(dataset_name):
    return [
        model_key.model_name
        for model_key in list_models(dataset_name, return_list=True, print_table=False)
        if model_key.dataset_name == dataset_name
    ]
