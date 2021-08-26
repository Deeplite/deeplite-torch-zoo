
from torchvision import models
from torch.hub import load_state_dict_from_url


__all__ = ["resnet18_tinyimagenet", "resnet34_tinyimagenet", "resnet50_tinyimagenet"]


model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18_tinyimagenet_0_663-b0637203dfcca31b.pt",
    "resnet34": "http://download.deeplite.ai/zoo/models/resnet34_tinyimagenet_0_6863-698d71e10fe153f0.pt",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50_tinyimagenet_0_7303-8ec06f70f32c110b.pt",
}


def resnet18_tinyimagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet18(num_classes=100)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["resnet18"], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def resnet34_tinyimagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet34(num_classes=100)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["resnet34"], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def resnet50_tinyimagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet50(num_classes=100)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["resnet50"], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)
    return model.to(device)
