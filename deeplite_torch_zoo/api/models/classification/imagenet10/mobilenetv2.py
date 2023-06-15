import torchvision
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mobilenet_v2_0_35_imagenet10"]

model_urls = {
    "mobilenetv2_0.35": "http://download.deeplite.ai/zoo/models/mobilenetv2_0.35-imagenet10-2410796e32dbde1c.pth",
    "mobilenetv2_1.0": "",
}


def _mobilenetv2_imagenet10(
    arch, alpha=1.0, pretrained=False, num_classes=10
):
    model = torchvision.models.mobilenet.MobileNetV2(
        width_mult=alpha, num_classes=num_classes
    )

    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenet_v2_0_35',
    dataset_name='imagenet10',
    task_type='classification',
)
def mobilenet_v2_0_35_imagenet10(
    pretrained=False, num_classes=10
):
    return _mobilenetv2_imagenet10(
        "mobilenetv2_0.35",
        alpha=0.35,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def mobilenet_v2_1_0(pretrained=False):
    return _mobilenetv2_imagenet10(
        "mobilenetv2_1.0",
        alpha=1,
        pretrained=pretrained,
    )
