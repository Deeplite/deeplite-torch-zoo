import torchvision

from deeplite_torch_zoo.utils.registry import MODEL_WRAPPER_REGISTRY


MODEL_NAMES = [
    "alexnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
    "squeezenet1_0",
    "squeezenet1_1",
    "inception_v3",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "googlenet",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]


def make_wrapper_func(wrapper_fn_name, model_name_key):
    @MODEL_WRAPPER_REGISTRY.register(model_name_key, 'imagenet','classification')
    def wrapper_func(pretrained=False, progress=True, device="cuda"):
        model = torchvision.models.__dict__[model_name_key](pretrained=pretrained)
        return model.to(device)

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in MODEL_NAMES:
    wrapper_name = "_".join((model_name_tag, "imagenet"))
    globals()[wrapper_name] = make_wrapper_func(wrapper_name, model_name_tag)
