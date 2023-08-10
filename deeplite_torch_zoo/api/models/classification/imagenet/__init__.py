from deeplite_torch_zoo.api.models.classification.imagenet.timm_models import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.imagenet.torchvision_models import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.imagenet.pytorchcv_models import *  # pylint: disable=unused-import
from deeplite_torch_zoo.api.models.classification.imagenet.zoo_models import *  # pylint: disable=unused-import

from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


REGISTER_PRIORITY = ('zoo', 'timm', 'torchvision', 'pytorchcv')

model_names = [
    model_key.model_name
    for model_key in MODEL_WRAPPER_REGISTRY.registry_dict
    if model_key.dataset_name == 'imagenet'
]
pretrained_model_names = [
    model_key.model_name
    for model_key in MODEL_WRAPPER_REGISTRY.pretrained_models
    if model_key.dataset_name == 'imagenet'
]
registered_model_names = []

# register models with clean names w/o source framework tag
for model_name in model_names:
    clean_model_name = '_'.join(model_name.split('_')[:-1])
    if clean_model_name in registered_model_names:
        continue
    for backend_tag in REGISTER_PRIORITY:
        backend_model_name = f'{clean_model_name}_{backend_tag}'
        if backend_model_name in model_names:
            wrapper_fn = MODEL_WRAPPER_REGISTRY.get(
                model_name=backend_model_name,
                dataset_name='imagenet',
            )
            has_checkpoint = backend_model_name in pretrained_model_names
            wrapper_fn = MODEL_WRAPPER_REGISTRY.register(
                model_name=clean_model_name,
                dataset_name='imagenet',
                task_type='classification',
                has_checkpoint=has_checkpoint,
            )(wrapper_fn)
            registered_model_names.append(clean_model_name)
            break
