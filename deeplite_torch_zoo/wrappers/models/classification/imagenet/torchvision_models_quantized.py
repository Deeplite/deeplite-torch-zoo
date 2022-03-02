import torchvision

from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


MODEL_NAMES = [
    "mobilenet_v2",
    "mobilenet_v3_large",
    "resnet18",
    "resnet50",
    "resnext101_32x8d",
    "googlenet",
    "inception_v3",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]


def make_wrapper_func(wrapper_fn_name, model_name_key):
    q_model_name = "_".join(("q", model_name_tag))
    @MODEL_WRAPPER_REGISTRY.register(model_name=q_model_name, dataset_name='imagenet', task_type='classification')
    def wrapper_func(pretrained=False, progress=True, device="cuda", num_classes=1000):
        model = torchvision.models.quantization.__dict__[model_name_key](pretrained=pretrained,
            num_classes=num_classes)
        return model.to(device)

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in MODEL_NAMES:
    wrapper_name = "_".join(("q", model_name_tag, "imagenet"))
    globals()[wrapper_name] = make_wrapper_func(wrapper_name, model_name_tag)
