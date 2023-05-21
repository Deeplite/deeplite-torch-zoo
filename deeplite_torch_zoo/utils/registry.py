# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Registry and RegistryStorage implementation modified from:
# https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/openvino/tools/pot/utils/registry.py
#

import re
from collections import namedtuple


def extract_model_type(model_name):
    MODEL_NAME_SUBSTRINGS = [
        'yolo',
    ]
    for substring in MODEL_NAME_SUBSTRINGS:
        if re.search(substring, model_name):
            return substring
    return model_name


def extract_dataset_type(dataset_name):
    DATASET_NAME_SUBSTRINGS = [
        'voc_format_dataset',
        'voc07',
        'voc',
        'coco',
        'wider_face',
        'lisa',
        'person_detection',
        'car_detection',
        'person_pet_vehicle_detection',
    ]
    DATASET_TYPE_MAP = {
        'car_detection': 'coco',
        'person_detection': 'voc',
        'person_pet_vehicle_detection': 'voc',
        'voc_format_dataset': 'voc',
    }
    for substring in DATASET_NAME_SUBSTRINGS:
        if re.search(substring, dataset_name):
            if substring in DATASET_TYPE_MAP:
                return DATASET_TYPE_MAP[substring]
            return substring
    return dataset_name


class Registry:
    def __init__(self, name=None, error_on_same_key=True):
        self._registry_dict = dict()
        self._name = name if name is not None else ''
        self._error_on_same_key = error_on_same_key

    def register(self, name=None, *args):
        def _register(obj_name, obj):
            if self._error_on_same_key and obj_name in self._registry_dict:
                raise KeyError(f'{obj_name} is already registered')
            self._registry_dict[obj_name] = obj

        def wrap(obj):
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            if args:
                cls_name = (cls_name, *args)
            _register(cls_name, obj)
            return obj

        return wrap

    def get(self, name):
        if name not in self._registry_dict:
            raise KeyError(f'{name} was not found in the {self._name} registry')
        return self._registry_dict[name]

    @property
    def registry_dict(self):
        return self._registry_dict

    @property
    def name(self):
        return self._name

    def __add__(self, other_registry):
        """
        Adding two objects of type Registry results in merging registry dicts
        """
        res = type(self)(self._name)
        if self._registry_dict.keys() & other_registry.registry_dict.keys():
            raise ValueError('Trying to add two registries with overlapping keys')
        res._registry_dict.update(self._registry_dict)
        res._registry_dict.update(other_registry.registry_dict)
        return res


class ModelWrapperRegistry(Registry):
    def __init__(self):
        super().__init__()
        self._task_type_map = dict()
        self._registry_key = namedtuple('RegistryKey', ['model_name', 'dataset_name'])
        self._registry_pretrained_models = {}

    @property
    def task_type_map(self):
        return self._task_type_map

    @property
    def pretrained_models(self):
        return self._registry_pretrained_models

    def register(self, model_name, dataset_name, task_type, has_checkpoint=True):
        def _register(obj_name, obj, task_type):
            if obj_name in self._registry_dict:
                raise KeyError(
                    f'{obj_name} is already registered in the model wrapper registry'
                )
            self._registry_dict[obj_name] = obj
            if has_checkpoint:
                self._registry_pretrained_models[obj_name] = obj
            self._task_type_map[obj_name] = task_type

        def wrap(obj):
            cls_name = model_name
            if cls_name is None:
                cls_name = obj.__name__
            cls_name = self._registry_key(
                model_name=cls_name, dataset_name=dataset_name
            )
            _register(cls_name, obj, task_type)
            return obj

        return wrap

    def get(self, model_name, dataset_name):
        key = self._registry_key(model_name=model_name, dataset_name=dataset_name)
        if key not in self._registry_dict:
            raise KeyError(
                f'Model {model_name} on dataset {dataset_name} was not found '
                'in the model wrapper registry'
            )
        return self._registry_dict[key]

    def get_task_type(self, model_name, dataset_name):
        GENERIC_DATASET_TASK_TYPE_MAP = {
            'voc_format_dataset': 'voc',
        }
        if dataset_name in GENERIC_DATASET_TASK_TYPE_MAP:
            dataset_name = GENERIC_DATASET_TASK_TYPE_MAP[dataset_name]
        key = self._registry_key(model_name=model_name, dataset_name=dataset_name)
        if key not in self._registry_dict:
            raise KeyError(
                f'Model {model_name} on dataset {dataset_name} was not found '
                'in the model wrapper registry'
            )
        return self._task_type_map[key]


class DatasetWrapperRegistry(Registry):
    def __init__(self):
        super().__init__()
        self._task_type_map = dict()
        self._registry_key = namedtuple('RegistryKey', ['dataset_name', 'model_type'])

    def register(self, dataset_name, model_type=None):
        def _register(obj_name, obj):
            if obj_name in self._registry_dict:
                raise KeyError(
                    f'{obj_name} is already registered in the datasplit wrapper registry'
                )
            self._registry_dict[obj_name] = obj

        def wrap(obj):
            cls_name = dataset_name
            if cls_name is None:
                cls_name = obj.__name__
            cls_name = self._registry_key(dataset_name=cls_name, model_type=model_type)
            _register(cls_name, obj)
            return obj

        return wrap

    def get(self, dataset_name, model_name):
        model_type = extract_model_type(model_name)
        key = self._registry_key(dataset_name=dataset_name, model_type=model_type)
        if key not in self._registry_dict:
            key = self._registry_key(dataset_name=dataset_name, model_type=None)
        if key not in self._registry_dict:
            raise KeyError(
                f'Dataset {dataset_name} for model type {model_type} was not found '
                'in the datasplit wrapper registry'
            )
        return self._registry_dict[key]


class EvaluatorWrapperRegistry(Registry):
    def __init__(self):
        super().__init__()
        self._task_type_map = dict()
        self._registry_key = namedtuple(
            'RegistryKey', ['dataset_type', 'model_type', 'task_type']
        )

    def register(self, task_type, model_type=None, dataset_type=None):
        def _register(obj_name, obj):
            if obj_name in self._registry_dict:
                raise KeyError(
                    f'{obj_name} is already registered in the evaluator wrapper registry'
                )
            self._registry_dict[obj_name] = obj

        def wrap(obj):
            cls_name = task_type
            if cls_name is None:
                cls_name = obj.__name__
            cls_name = self._registry_key(
                task_type=cls_name, model_type=model_type, dataset_type=dataset_type
            )
            _register(cls_name, obj)
            return obj

        return wrap

    def get(self, task_type, model_name, dataset_name):
        model_type = extract_model_type(model_name)
        dataset_type = extract_dataset_type(dataset_name)

        key = self._registry_key(
            task_type=task_type, dataset_type=dataset_type, model_type=model_type
        )
        if key not in self._registry_dict:
            key = self._registry_key(
                task_type=task_type, model_type=model_type, dataset_type=None
            )

        if key not in self._registry_dict:
            key = self._registry_key(
                task_type=task_type, model_type=None, dataset_type=None
            )

        if key not in self._registry_dict:
            raise KeyError(
                f'{task_type} model {model_name} on dataset {dataset_name} was not found '
                'in the evaluator wrapper registry'
            )

        return self._registry_dict[key]


class RegistryStorage:
    def __init__(self, registry_list):
        regs = [r for r in registry_list if isinstance(r, Registry)]
        self.registries = {}
        for r in regs:
            if r.name in self.registries:
                raise RuntimeError(
                    f'There are more than one registry with the name "{r.name}"'
                )
            self.registries[r.name] = r

    def get(self, registry_name: str) -> Registry:
        if registry_name not in self.registries:
            raise RuntimeError(
                f'Cannot find registry with registry_name "{registry_name}"'
            )
        return self.registries[registry_name]
