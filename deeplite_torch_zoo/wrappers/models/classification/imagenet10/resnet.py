import torchvision
from torch.hub import load_state_dict_from_url

__all__ = ["resnet18_imagenet10"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-imagenet10-f119488aa5e047b0.pth",
}


def _resnet_imagenet10(arch, pretrained=False, progress=True, device='cuda'):
    model = torchvision.models.resnet18(num_classes=10)

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def resnet18_imagenet10(pretrained=False, progress=True, device='cuda'):
    return _resnet_imagenet10("resnet18", pretrained=pretrained, progress=progress, device=device)
