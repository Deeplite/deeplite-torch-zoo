import torchvision
from torch.hub import load_state_dict_from_url

__all__ = [
    # 'mobilenet_v2_1_0',
    "mobilenet_v2_0_35_imagenet10"
]

model_urls = {
    "mobilenetv2_0.35": "http://download.deeplite.ai/zoo/models/mobilenetv2_0.35-imagenet10-2410796e32dbde1c.pth",
    "mobilenetv2_1.0": "",
}


def _mobilenetv2_imagenet10(arch, alpha=1.0, pretrained=False, progress=True, device='cuda'):
    model = torchvision.models.mobilenet.MobileNetV2(width_mult=alpha, num_classes=10)

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def mobilenet_v2_0_35_imagenet10(pretrained=False, progress=True, device='cuda'):
    return _mobilenetv2_imagenet10(
        "mobilenetv2_0.35", alpha=0.35, pretrained=pretrained, progress=progress, device=device
    )


def mobilenet_v2_1_0(pretrained=False, progress=True, device='cuda'):
    return _mobilenetv2_imagenet10(
        "mobilenetv2_1.0", alpha=1, pretrained=pretrained, progress=progress, device=device
    )
