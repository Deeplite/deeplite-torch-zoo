import timm
from deeplite_torch_zoo.wrappers.models.classification.imagenet.torchvision_models import MODEL_NAMES


TIMM_MODEL_NAMES = set(timm.list_models()) - set(MODEL_NAMES)

def make_wrapper_func(wrapper_fn_name, model_name_key):
    def wrapper_func(pretrained=False, progress=True, device="cuda"):
        model = timm.create_model(model_name_key, pretrained=pretrained)
        return model.to(device)

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in TIMM_MODEL_NAMES:
    wrapper_name = "_".join((model_name_tag, "imagenet"))
    globals()[wrapper_name] = make_wrapper_func(wrapper_name, model_name_tag)
