import torchvision

from deeplite_torch_zoo.utils import load_state_dict_partial, LOGGER
from deeplite_torch_zoo.api.models.classification.model_implementation_dict import (
    MODEL_IMPLEMENTATIONS,
)
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.api.models.classification.imagenet.utils import NUM_IMAGENET_CLASSES


def make_wrapper_func(wrapper_fn_name, register_model_name_key, model_name_key):
    @MODEL_WRAPPER_REGISTRY.register(
        model_name=register_model_name_key,
        dataset_name='imagenet',
        task_type='classification',
    )
    def wrapper_func(
        pretrained=False, num_classes=NUM_IMAGENET_CLASSES, **model_kwargs
    ):
        LOGGER.warning(f'Extra options {model_kwargs} are not applicable for a torchvision model')
        model = torchvision.models.__dict__[model_name_key](
            pretrained=pretrained, num_classes=NUM_IMAGENET_CLASSES
        )

        if num_classes != NUM_IMAGENET_CLASSES:
            pretrained_dict = model.state_dict()
            model = torchvision.models.__dict__[model_name_key](
                pretrained=False, num_classes=num_classes
            )
            load_state_dict_partial(model, pretrained_dict)

        return model

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in MODEL_IMPLEMENTATIONS['torchvision']:
    register_model_name_tag = f'{model_name_tag}_torchvision'
    wrapper_name = '_'.join((model_name_tag, 'imagenet'))
    globals()[wrapper_name] = make_wrapper_func(
        wrapper_name,
        register_model_name_key=register_model_name_tag,
        model_name_key=model_name_tag,
    )
