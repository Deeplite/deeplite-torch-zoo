from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.classification.mobilenetv3 import MobileNetV3

__all__ = ["mobilenet_v3_vww"]

model_urls = {
    "mobilenetv3": "http://download.deeplite.ai/zoo/models/mobilenetv3-vww-1d2be1e7d5473081.pth",
}


def _mobilenetv3_vww(arch, pretrained=False, progress=True, device='cuda'):
    model = MobileNetV3(num_classes=2)

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def mobilenet_v3_vww(pretrained=False, progress=True, device='cuda'):
    return _mobilenetv3_vww("mobilenetv3", pretrained=pretrained, progress=progress, device=device)
