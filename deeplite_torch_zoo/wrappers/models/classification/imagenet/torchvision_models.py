import timm
import torchvision
from deeplite_torch_zoo.wrappers.models.classification.imagenet.impl_model_names import \
    IMPL_MODEL_NAMES
from deeplite_torch_zoo.wrappers.models.utils import load_state_dict_partial
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

TIMM_MODELS = timm.list_models()
NUM_IMAGENET_CLASSES = 1000

def make_wrapper_func(wrapper_fn_name, register_model_name_key, model_name_key):
    @MODEL_WRAPPER_REGISTRY.register(model_name=register_model_name_key, dataset_name='imagenet', task_type='classification')
    @MODEL_WRAPPER_REGISTRY.register(model_name=register_model_name_key, dataset_name='food101',
        task_type='classification', has_checkpoint=False)
    def wrapper_func(pretrained=False, progress=True, device="cuda", num_classes=NUM_IMAGENET_CLASSES):
        model = torchvision.models.__dict__[model_name_key](pretrained=pretrained, num_classes=NUM_IMAGENET_CLASSES)

        if num_classes != NUM_IMAGENET_CLASSES:
            pretrained_dict = model.state_dict()
            model = torchvision.models.__dict__[model_name_key](pretrained=False, num_classes=num_classes)
            load_state_dict_partial(model, pretrained_dict)

        return model.to(device)

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in IMPL_MODEL_NAMES['torchvision']:
    register_model_name_tag = model_name_tag
    if model_name_tag in TIMM_MODELS:
        register_model_name_tag = "_".join((model_name_tag, "torchvision"))
    wrapper_name = "_".join((model_name_tag, "imagenet"))
    globals()[wrapper_name] = make_wrapper_func(wrapper_name,
        register_model_name_key=register_model_name_tag, model_name_key=model_name_tag)
