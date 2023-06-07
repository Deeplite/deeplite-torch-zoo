import timm
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.api.models.classification.imagenet.utils import NUM_IMAGENET_CLASSES


def make_wrapper_func(wrapper_fn_name, model_name_key):
    @MODEL_WRAPPER_REGISTRY.register(
        model_name=f'{model_name_key}_timm', dataset_name='imagenet', task_type='classification'
    )
    def wrapper_func(pretrained=False, device="cuda", num_classes=NUM_IMAGENET_CLASSES):
        model = timm.create_model(
            model_name_key, pretrained=pretrained, num_classes=num_classes
        )
        return model.to(device)

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in timm.list_models():
    wrapper_name = '_'.join((model_name_tag, 'imagenet'))
    globals()[wrapper_name] = make_wrapper_func(wrapper_name, model_name_tag)
