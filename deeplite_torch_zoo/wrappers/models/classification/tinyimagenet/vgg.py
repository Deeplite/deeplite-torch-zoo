
from torchvision import models
from torch.hub import load_state_dict_from_url


__all__ = ["vgg19_tinyimagenet"]


model_urls = {
    "vgg19": "http://download.deeplite.ai/zoo/models/vgg19_tinyimagenet_0-7288-aaa20280ea9bb886.pt",
}


def vgg19_tinyimagenet(pretrained=False, progress=True, device="cuda"):
    model = models.vgg19(num_classes=100)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["vgg19"], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)
    return model.to(device)
