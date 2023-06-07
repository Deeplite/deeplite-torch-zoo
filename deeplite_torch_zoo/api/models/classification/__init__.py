import functools

from deeplite_torch_zoo.api.models.classification.imagenet import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.cifar100 import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.flowers102 import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.imagenet10 import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.imagenet16 import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.mnist import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.tinyimagenet import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.vww import *  # pylint: disable=unused-import


EXTRA_CLS_DATASETS = {
    'food101': 101,
    'imagewoof': 10,
    'imagenette': 10,
}

imagenet_model_names = [model_key.model_name for model_key in MODEL_WRAPPER_REGISTRY.registry_dict
                        if model_key.dataset_name == 'imagenet']

# register models for some extra datasets
for model_name in imagenet_model_names:
    has_checkpoint = False
    if model_name in FLOWERS101_CHECKPOINT_URLS:
        has_checkpoint = True
    for dataset_name, num_classes in EXTRA_CLS_DATASETS.items():
        wrapper_name = '_'.join((model_name, dataset_name))
        wrapper_fn = MODEL_WRAPPER_REGISTRY.get(
            model_name=model_name,
            dataset_name='imagenet',
        )
        wrapper_fn = MODEL_WRAPPER_REGISTRY.register(
            model_name=model_name,
            dataset_name=dataset_name,
            task_type='classification',
            has_checkpoint=has_checkpoint,
        )(functools.partial(wrapper_fn, num_classes=num_classes))
