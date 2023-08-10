import torchvision
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["resnet18_imagenet10"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-imagenet10-f119488aa5e047b0.pth",
}


def _resnet_imagenet10(
    arch, pretrained=False, num_classes=10
):
    model = torchvision.models.resnet18(num_classes=num_classes)

    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model


@MODEL_WRAPPER_REGISTRY.register(
    model_name='resnet18', dataset_name='imagenet10', task_type='classification'
)
def resnet18_imagenet10(pretrained=False, num_classes=10):
    return _resnet_imagenet10(
        "resnet18",
        pretrained=pretrained,
        num_classes=num_classes,
    )
