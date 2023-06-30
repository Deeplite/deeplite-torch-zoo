from deeplite_torch_zoo.src.classification.mobilenets.mobilenetv3 import (
    mobilenetv3_large,
    mobilenetv3_small,
)
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY

__all__ = ["mobilenetv3_small_vww", "mobilenetv3_large_vww"]

model_urls = {
    "mobilenetv3_small": "http://download.deeplite.ai/zoo/models/mobilenetv3-small-vww-89_20-5224256355d8fbfa.pth",
    "mobilenetv3_large": "http://download.deeplite.ai/zoo/models/mobilenetv3-large-vww-89_14-e80487ebdbb41d5a.pth",
}


def _mobilenetv3_vww(
    arch="small", pretrained=False, num_classes=2
):
    if arch == "small":
        model = mobilenetv3_small(num_classes=num_classes)
    elif arch == "large":
        model = mobilenetv3_large(num_classes=num_classes)

    if pretrained:
        checkpoint_url = model_urls[f"mobilenetv3_{arch}"]
        model = load_pretrained_weights(model, checkpoint_url)

    return model


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenetv3_small', dataset_name='vww', task_type='classification'
)
def mobilenetv3_small_vww(
    pretrained=False, num_classes=2
):
    return _mobilenetv3_vww(
        arch="small",
        pretrained=pretrained,
        num_classes=num_classes,
    )


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenetv3_large', dataset_name='vww', task_type='classification'
)
def mobilenetv3_large_vww(
    pretrained=False, num_classes=2,
):
    return _mobilenetv3_vww(
        arch="large",
        pretrained=pretrained,
        num_classes=num_classes,
    )
