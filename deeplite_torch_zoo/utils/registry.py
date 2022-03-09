# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import re

def extract_model_type(net):
    MODEL_NAME_SUBSTRINGS = ["yolo", "unet_scse", "unet", "ssd300", "deeplab",
                            "fcn", "mb2_ssd_lite","mb2_ssd", "mb1_ssd", "ssd", "rcnn"]
    for substring in MODEL_NAME_SUBSTRINGS:
        if re.search(substring, net):
            return substring
    return net

def normalize_dataset_name(data):
    if 'voc07' in data:
        data = 'voc07'
    elif 'voc' in data:
        data = 'voc'
    elif 'coco' in data:
        data = 'coco'
    elif 'wider_face' in data:
        data = 'wider_face'
    elif 'lisa' in data:
        data = 'lisa'
    elif 'person_detection' in data:
        data = 'voc'
    elif 'person_pet_vehicle_detection':
        data = 'voc'
    return data

class Registry:
    """ Generic registry implentation modified from
    https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/openvino/tools/pot/utils/registry.py """

    def __init__(self):
        self._registry_dict = dict()

    def register(self, name=None, *args):
        def _register(obj_name, obj):
            if obj_name in self._registry_dict:
                raise KeyError(f'{obj_name} is already registered')
            self._registry_dict[obj_name] = obj

        def wrap(obj):
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            cls_name = (cls_name, *args)
            _register(cls_name, obj)
            return obj

        return wrap

    def get(self, name):
        if name not in self._registry_dict:
            raise KeyError(f'{name} was not found in the registry')
        return self._registry_dict[name]

    @property
    def registry_dict(self):
        return self._registry_dict

    @property
    def name(self):
        return self._name


class ModelWrapperRegistry(Registry):

    def __init__(self):
        super().__init__()
        self._task_type_map = dict()
        self._registry_key = namedtuple('RegistryKey', ['model_name', 'dataset_name'])

    @property
    def task_type_map(self):
        return self._task_type_map

    def register(self, model_name, dataset_name, task_type):
        def _register(obj_name, obj, task_type):
            if obj_name in self._registry_dict:
                raise KeyError(f'{obj_name} is already registered in the model wrapper registry')
            self._registry_dict[obj_name] = obj
            self._task_type_map[obj_name] = task_type

        def wrap(obj):
            cls_name = model_name
            if cls_name is None:
                cls_name = obj.__name__
            cls_name = self._registry_key(model_name=cls_name, dataset_name=dataset_name)
            _register(cls_name, obj, task_type)
            return obj

        return wrap

    def get(self, model_name, dataset_name):
        key = self._registry_key(model_name=model_name, dataset_name=dataset_name)
        if key not in self._registry_dict:
            raise KeyError(f'Model {model_name} on dataset {dataset_name} was not found '
                'in the model wrapper registry')
        return self._registry_dict[key]

class EvalWrapperRegistry(Registry):

    def __init__(self):
        super().__init__()
    
    def register(self, name=None):
        def _register(obj_name, obj):
            if obj_name in self._registry_dict:
                raise KeyError(f'{obj_name} is already registered in the model wrapper registry')
            self._registry_dict[obj_name] = obj

        def wrap(obj):
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            cls_name = (cls_name)
            _register(cls_name, obj)
            return obj

        return wrap

    def get(self, task_type, model_name, dataset_name):
        if task_type == "classification":
            key = "classification"
        elif task_type == "semantic_segmentation":
            model_type = extract_model_type(model_name)
            key = task_type +"_"+ model_type
        elif task_type == 'object_detection':
            model_type = extract_model_type(model_name)
            dataset_type = normalize_dataset_name(dataset_name)
            key = task_type + "_" + model_type + "_" + dataset_type
        
        if key not in self._registry_dict:
            raise KeyError(f'{task_type} Model {model_name} on dataset {dataset_name} was not found '
                'in the eval wrapper registry')
        return self._registry_dict[key]