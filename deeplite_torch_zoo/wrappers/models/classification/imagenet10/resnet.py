import torchvision
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["resnet18_imagenet10"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-imagenet10-f119488aa5e047b0.pth",
}


def _resnet_imagenet10(arch, pretrained=False, progress=True, num_classes=10, device='cuda'):
    model = torchvision.models.resnet18(num_classes=num_classes)

    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet18', dataset_name='imagenet10', task_type='classification')
def resnet18_imagenet10(pretrained=False, progress=True, num_classes=10, device='cuda'):
    return _resnet_imagenet10("resnet18", pretrained=pretrained, progress=progress, num_classes=num_classes, device=device)
