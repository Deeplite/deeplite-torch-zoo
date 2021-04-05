import torchvision
from torch.hub import load_state_dict_from_url

__all__ = ["resnet18_imagenet16", "resnet50_imagenet16"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-imagenet16-2f8c56bafc30cde9.pth",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50-imagenet16-f546a9fdf7bff1b9.pth",
}


def _resnet_imagenet16(arch, pretrained=False, progress=True, device='cuda'):
    if arch == "resnet18":
        model = torchvision.models.resnet18(num_classes=16)
    elif arch == "resnet50":
        model = torchvision.models.resnet50(num_classes=16)
    else:
        raise ValueError

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def resnet18_imagenet16(pretrained=False, progress=True, device='cuda'):
    return _resnet_imagenet16("resnet18", pretrained=pretrained, progress=progress, device=device)


def resnet50_imagenet16(pretrained=False, progress=True, device='cuda'):
    return _resnet_imagenet16("resnet50", pretrained=pretrained, progress=progress, device=device)
