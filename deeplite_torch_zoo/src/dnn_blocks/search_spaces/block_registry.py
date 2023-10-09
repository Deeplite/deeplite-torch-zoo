import itertools
from functools import partial

from deeplite_torch_zoo.utils.registry import Registry


class DNNBlockRegistry(Registry):
    def __init__(self, name):
        super().__init__(name)
        self._registry_dict_block_name = {}

    @property
    def block_name_dict(self):
        return self._registry_dict_block_name

    def __add__(self, other_registry):
        """
        Adding two objects of type Registry results in merging registry dicts
        """
        res = type(self)(self._name)
        if self._registry_dict.keys() & other_registry.registry_dict.keys():
            raise ValueError('Trying to add two registries with overlapping keys')

        res._registry_dict.update(self._registry_dict)
        res._registry_dict.update(other_registry.registry_dict)

        res._registry_dict_block_name.update(self._registry_dict_block_name)
        res._registry_dict_block_name.update(other_registry._registry_dict_block_name)

        return res

    def register(self, name: str = None, **kwargs):
        def _register(obj_name, obj):
            registry_key = str(obj_name)
            if registry_key in self._registry_dict:
                raise KeyError(f'{registry_key} is already registered')

            self._registry_dict[registry_key] = obj
            self._registry_dict_block_name[registry_key] = obj_name

        def wrap(obj):
            cls_name = name
            if kwargs:
                # Register objects with all combinations of kwarg values
                combinations = itertools.product(*list(kwargs.values()))
                for combination in combinations:
                    cls_name = name
                    if cls_name is None:
                        cls_name = obj.__name__
                    obj_kwargs = dict(zip(list(kwargs.keys()), combination))
                    new_obj = partial(obj, **obj_kwargs)
                    cls_name = (cls_name, tuple(obj_kwargs.items()))
                    _register(cls_name, new_obj)
            else:
                if cls_name is None:
                    cls_name = obj.__name__
                cls_name = (cls_name, {})
                _register(cls_name, obj)
            return obj

        return wrap
