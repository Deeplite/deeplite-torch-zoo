import torchvision
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["resnet18_imagenet16", "resnet50_imagenet16"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-imagenet16-2f8c56bafc30cde9.pth",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50-imagenet16-f546a9fdf7bff1b9.pth",
}


def _resnet_imagenet16(arch, pretrained=False, progress=True, num_classes=16, device='cuda'):
    if arch == "resnet18":
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif arch == "resnet50":
        model = torchvision.models.resnet50(num_classes=num_classes)
    else:
        raise ValueError

    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet18', dataset_name='imagenet16', task_type='classification')
def resnet18_imagenet16(pretrained=False, progress=True, num_classes=16, device='cuda'):
    return _resnet_imagenet16("resnet18", pretrained=pretrained, progress=progress, device=device, num_classes=num_classes)


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet50', dataset_name='imagenet16', task_type='classification')
def resnet50_imagenet16(pretrained=False, progress=True, num_classes=16, device='cuda'):
    return _resnet_imagenet16("resnet50", pretrained=pretrained, progress=progress, device=device, num_classes=num_classes)
