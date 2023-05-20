import timm
import torchvision
from pytorchcv.model_provider import get_model as ptcv_get_model

from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.models.classification.flowers102.model_urls import \
    model_urls
from deeplite_torch_zoo.api.models.classification.impl_model_names import \
    IMPL_MODEL_NAMES
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY

CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/flowers102"


def find_model_implementation(model_name, num_classes):
    if model_name in timm.list_models():
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if model_name.replace('_pytorchcv', '') in IMPL_MODEL_NAMES['pytorchcv']:
        return ptcv_get_model(model_name.replace('_pytorchcv', ''), num_classes=num_classes)
    if model_name.replace('_torchvision', '') in IMPL_MODEL_NAMES['torchvision']:
        return torchvision.models.__dict__[model_name.replace('_torchvision', '')](pretrained=False, num_classes=num_classes)
    raise ValueError(f'Can\'t find model implementation of {model_name} in the zoo')


def make_wrapper_func(wrapper_fn_name, model_name_key):
    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name_key, dataset_name='flowers102',
        task_type='classification')
    def wrapper_func(pretrained=False, progress=True, device='cuda', num_classes=102):

        model = find_model_implementation(model_name_key, num_classes=num_classes)

        if pretrained:
            checkpoint_url = f'{CHECKPOINT_STORAGE_URL}/{model_urls[model_name_key]}'
            model = load_pretrained_weights(model, checkpoint_url, progress, device)

        return model.to(device)

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in model_urls:
    wrapper_name = '_'.join((model_name_tag, 'flowers102'))
    globals()[wrapper_name] = make_wrapper_func(wrapper_name, model_name_tag)
