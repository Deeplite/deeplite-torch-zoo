import timm
from deeplite_torch_zoo.wrappers.models.classification.imagenet.impl_model_names import \
    IMPL_MODEL_NAMES
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY
from pytorchcv.model_provider import get_model as ptcv_get_model

TIMM_MODELS = timm.list_models()


def make_wrapper_func(wrapper_fn_name, register_model_name_key, model_name_key):
    @MODEL_WRAPPER_REGISTRY.register(model_name=register_model_name_key, dataset_name='imagenet', task_type='classification')
    @MODEL_WRAPPER_REGISTRY.register(model_name=register_model_name_key, dataset_name='food101',
        task_type='classification', has_checkpoint=False)
    def wrapper_func(pretrained=False, progress=True, device="cuda", num_classes=1000):
        model = ptcv_get_model(model_name_key, pretrained=pretrained, num_classes=num_classes)
        return model.to(device)

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in IMPL_MODEL_NAMES['pytorchcv']:
    register_model_name_tag = model_name_tag
    if model_name_tag in TIMM_MODELS:
        register_model_name_tag = "_".join((model_name_tag, "pytorchcv"))
    wrapper_name = "_".join((model_name_tag, "imagenet"))
    globals()[wrapper_name] = make_wrapper_func(wrapper_name,
        register_model_name_key=register_model_name_tag, model_name_key=model_name_tag)
