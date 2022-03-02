import torchvision
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["resnet18_vww", "resnet50_vww"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-vww-7f02ab4b50481ab7.pth",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50-vww-9d4cb2cb19f8c5d5.pth",
}


def _resnet_vww(arch, pretrained=False, progress=True, num_classes=2, device='cuda'):
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


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet18', dataset_name='vww', task_type='classification')
def resnet18_vww(pretrained=False, progress=True, num_classes=2, device='cuda'):
    return _resnet_vww("resnet18", pretrained=pretrained, num_classes=num_classes, progress=progress, device=device)


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet50', dataset_name='vww', task_type='classification')
def resnet50_vww(pretrained=False, progress=True, num_classes=2, device='cuda'):
    return _resnet_vww("resnet50", pretrained=pretrained, num_classes=num_classes, progress=progress, device=device)
