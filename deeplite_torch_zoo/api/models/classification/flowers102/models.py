from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.models.classification.flowers102.model_urls import (
    FLOWERS101_CHECKPOINT_URLS,
)
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY

CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/flowers102"
FLOWERS102_NUM_CLASSES = 102

imagenet_model_names = [
    model_key.model_name
    for model_key in MODEL_WRAPPER_REGISTRY.registry_dict
    if model_key.dataset_name == 'imagenet'
]


def make_wrapper_func(wrapper_fn_name, model_name_key, has_checkpoint):
    @MODEL_WRAPPER_REGISTRY.register(
        model_name=model_name_key,
        dataset_name='flowers102',
        task_type='classification',
        has_checkpoint=has_checkpoint,
    )
    def wrapper_func(
        pretrained=False, num_classes=FLOWERS102_NUM_CLASSES
    ):
        wrapper_fn = MODEL_WRAPPER_REGISTRY.get(
            model_name=model_name_key, dataset_name='imagenet'
        )
        model = wrapper_fn(
            pretrained=False,
            num_classes=num_classes,
        )
        if pretrained:
            checkpoint_url = (
                f'{CHECKPOINT_STORAGE_URL}/{FLOWERS101_CHECKPOINT_URLS[model_name_key]}'
            )
            model = load_pretrained_weights(model, checkpoint_url)

        return model

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in imagenet_model_names:
    has_pretrained_checkpoint = False
    wrapper_name = '_'.join((model_name_tag, 'flowers102'))
    if model_name_tag in FLOWERS101_CHECKPOINT_URLS:
        has_pretrained_checkpoint = True
    globals()[wrapper_name] = make_wrapper_func(
        wrapper_name, model_name_tag, has_pretrained_checkpoint
    )
