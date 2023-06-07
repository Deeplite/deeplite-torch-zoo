from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.models.classification.flowers102.model_urls import (
    model_urls,
)
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY

CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/flowers102"
FLOWERS102_NUM_CLASSES = 102


def make_wrapper_func(wrapper_fn_name, model_name_key):
    @MODEL_WRAPPER_REGISTRY.register(
        model_name=model_name_key, dataset_name='flowers102', task_type='classification'
    )
    def wrapper_func(pretrained=False, device='cuda', num_classes=FLOWERS102_NUM_CLASSES):
        wrapper_fn = MODEL_WRAPPER_REGISTRY.get(model_name=model_name_key, dataset_name='imagenet')
        model = wrapper_fn(
            pretrained=False, num_classes=num_classes, device=device,
        )
        if pretrained:
            checkpoint_url = f'{CHECKPOINT_STORAGE_URL}/{model_urls[model_name_key]}'
            model = load_pretrained_weights(model, checkpoint_url, device)

        return model.to(device)

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in model_urls:
    wrapper_name = '_'.join((model_name_tag, 'flowers102'))
    globals()[wrapper_name] = make_wrapper_func(wrapper_name, model_name_tag)
