import fnmatch
import collections
import texttable

import sys

import deeplite_torch_zoo.wrappers.datasets  # pylint: disable=unused-import
import deeplite_torch_zoo.wrappers.models  # pylint: disable=unused-import
import deeplite_torch_zoo.wrappers.eval # pylint: disable=unused-import
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.wrappers.registries import DATA_WRAPPER_REGISTRY
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY


__all__ = ["get_data_splits_by_name", "get_model_by_name",
    "list_models", "get_eval_function"]


def normalize_model_name(net):
    if "yolo" in net:
        return "yolo"
    if "unet" in net:
        return "unet"
    if "ssd300" in net:
        return "ssd300"
    return net


def get_eval_querry(model_name, dataset_name):

    if ('yolo' in model_name):
        querry_1 = 'object_detection'
        querry_2 = 'yolo'
        if ('voc07' in dataset_name):
            querry_3 = 'voc07'
            return [querry_1, querry_2, querry_3]
        elif (('voc'in dataset_name) or ('person_detection' in dataset_name) or ('person_pet_vehicle_detection' in dataset_name)):
            querry_3 = 'voc'
            return [querry_1, querry_2, querry_3]
        elif ('lisa' in dataset_name):
            querry_3 = 'lisa'
            return [querry_1, querry_2, querry_3]
        elif ('car_detection' in dataset_name):
            querry_3 = 'car_detection'
            return [querry_1, querry_2, querry_3]
        elif ('wider_face' in dataset_name):
            querry_3 = 'wider_face'
            return [querry_1, querry_2, querry_3]
        elif ('coco' in dataset_name):
            querry_3 = 'coco'
            return [querry_1, querry_2, querry_3]
        
        
    elif ("ssd" in model_name):
        querry_1 = 'object_detection'
        querry_2 = model_name
        querry_3 = 'general_obj_detect_dataset'
        return [querry_1, querry_2, querry_3]

    elif ("rcnn" in model_name):
        querry_1 = 'object_detection'
        querry_2 = model_name
        querry_3 = dataset_name
        return [querry_1, querry_2, querry_3]
    
    elif ("unet" in model_name):
        querry_1 = 'segmentation'
        querry_3 = 'general_seg_dataset'
        if ("unet_scse" in model_name):
            querry_2 = "unet_scse"
        else:
            querry_2 = "unet"
        return [querry_1, querry_2, querry_3]

    elif ("fcn" in model_name):
        querry_1 = 'segmentation'
        querry_2 = 'fcn'
        querry_3 = 'general_seg_dataset'
        return [querry_1, querry_2, querry_3]
    
    elif ('deeplab' in model_name):
        querry_1 = 'segmentation'
        querry_2 = 'deeplab'
        querry_3 = 'general_seg_dataset'
        return [querry_1, querry_2, querry_3]

    else:
        querry_1 = 'classification'
        querry_2 = 'general_classification_model'
        querry_3 = 'general_classification_dataset'
        return [querry_1, querry_2, querry_3]




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
    model_func = MODEL_WRAPPER_REGISTRY.get((model_name.lower(), dataset_name))
    model = model_func(pretrained=pretrained, progress=progress, device=device)

    return model.half() if fp16 else model



def get_eval_function(model_name="",dataset_name=""):

    """
    Tries to find a matching model creation fn in the registry and identify the task type
    using model name and dataset name.
    :param model_name: Name of the model to create
    :param dataset_name: Name of dataset the model was trained / is to be trained on

    returns a corresponding evaluation funcgion
    """

    task_type, model, data_name = get_eval_querry(model_name.lower(), dataset_name)
    eval_func = EVAL_WRAPPER_REGISTRY.get((task_type, model, data_name))

    return eval_func


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
