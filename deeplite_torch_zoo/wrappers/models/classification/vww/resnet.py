import torchvision
from torch.hub import load_state_dict_from_url

__all__ = ["resnet18_vww", "resnet50_vww"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-vww-7f02ab4b50481ab7.pth",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50-vww-9d4cb2cb19f8c5d5.pth",
}


def _resnet_vww(arch, pretrained=False, progress=True, device='cuda'):
    if arch == "resnet18":
        model = torchvision.models.resnet18(num_classes=2)
    elif arch == "resnet50":
        model = torchvision.models.resnet50(num_classes=2)
    else:
        raise ValueError

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def resnet18_vww(pretrained=False, progress=True, device='cuda'):
    return _resnet_vww("resnet18", pretrained=pretrained, progress=progress, device=device)


def resnet50_vww(pretrained=False, progress=True, device='cuda'):
    return _resnet_vww("resnet50", pretrained=pretrained, progress=progress, device=device)
