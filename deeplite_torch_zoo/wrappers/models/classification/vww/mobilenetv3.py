from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.classification.mobilenetv3 import mobilenetv3_large, mobilenetv3_small

__all__ = ["mobilenetv3_small_vww", "mobilenetv3_large_vww"]

model_urls = {
    "mobilenetv3_small": "http://download.deeplite.ai/zoo/models/mobilenetv3-small-vww-89_20-5224256355d8fbfa.pth",
    "mobilenetv3_large": "http://download.deeplite.ai/zoo/models/mobilenetv3-large-vww-89_14-e80487ebdbb41d5a.pth",
}


def _mobilenetv3_vww(arch="small", pretrained=False, progress=True, device='cuda'):
    if arch == "small":
        model = mobilenetv3_small(num_classes=2)
    elif arch == "large":
        model = mobilenetv3_large(num_classes=2)


    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[f"mobilenetv3_{arch}"], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def mobilenetv3_small_vww(pretrained=False, progress=True, device='cuda'):
    return _mobilenetv3_vww(arch="small", pretrained=pretrained, progress=progress, device=device)


def mobilenetv3_large_vww(pretrained=False, progress=True, device='cuda'):
    return _mobilenetv3_vww(arch="large", pretrained=pretrained, progress=progress, device=device)
