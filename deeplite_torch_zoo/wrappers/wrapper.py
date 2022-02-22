import fnmatch
import collections

import texttable

import deeplite_torch_zoo.wrappers.datasets  # pylint: disable=unused-import
import deeplite_torch_zoo.wrappers.models  # pylint: disable=unused-import
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.wrappers.registries import DATA_WRAPPER_REGISTRY



__all__ = ["get_data_splits_by_name", "get_model_by_name",
    "list_models"]


def normalize_model_name(net):
    if "yolo" in net:
        return "yolo"
    if "unet" in net:
        return "unet"
    if "ssd300" in net:
        return "ssd300"
    return net


def get_data_splits_by_name(data_root="", dataset_name="", model_name=None, **kwargs):
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
    datasplit_key = (dataset_name.lower(), )
    if model_name is not None:
        model_name = normalize_model_name(model_name)
        model_name = model_name.lower()
        datasplit_key += (model_name, )

    data_split_wrapper_fn = DATA_WRAPPER_REGISTRY.get(datasplit_key)
    data_split = data_split_wrapper_fn(data_root=data_root, **kwargs)
    return data_split


def get_model_by_name(
    model_name="", dataset_name="", pretrained=False, progress=False, fp16=False, device="cuda"
):
    """
    Tries to find a matching model creation fn in the registry and creates a new model object
    :param model_name: Name of the model to create
    :param dataset_name: Name of dataset the model was trained / is to be trained on
    :param pretrained: Whether to load pretrained weights
    :param progress: Whether to enable the progressbar
    :param fp16: Whether to convert the model to fp16 precision
    :param device: Loads the model either on a gpu (`cuda`, `cuda:device_id`) or cpu.

    returns a corresponding model object (optionally with pretrained weights)
    """
    model_func = MODEL_WRAPPER_REGISTRY.get(model_name=model_name.lower(), dataset_name=dataset_name)
    model = model_func(pretrained=pretrained, progress=progress, device=device)
    return model.half() if fp16 else model


def list_models(filter='', print_table=True, return_list=False):
    """
    A helper function to list all existing models or dataset calls
    It takes a `model_name` or a `dataset_name` as a filter and
    prints a table of corresponding available models
    """
    filter = '*' + filter + '*'
    all_models = MODEL_WRAPPER_REGISTRY.registry_dict.keys()
    all_models = [(model_name, dataset_name) for model_name, dataset_name in all_models
        if dataset_name is not None]
    all_models = {model_name + '_' + dataset_name: (model_name, dataset_name)
        for model_name, dataset_name in all_models}

    models = []
    include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
    for f in include_filters:
        include_models = fnmatch.filter(all_models.keys(), f)
        if include_models:
            models = set(models).union(include_models)

    model_dataset_pairs = [all_models[model_key] for model_key in sorted(models)]

    if print_table:
        table = texttable.Texttable()
        rows = collections.defaultdict(list)
        for model, dataset in model_dataset_pairs:
            rows[model].extend([dataset])
        for model in rows:
            rows[model] = ', '.join(rows[model])
        table.add_rows([['Available models', 'Source datasets'], *rows.items()])
        print(table.draw())

    if return_list:
        return model_dataset_pairs
    return None
