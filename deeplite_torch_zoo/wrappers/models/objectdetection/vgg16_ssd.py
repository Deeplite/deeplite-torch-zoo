from torch.hub import load_state_dict_from_url

__all__ = ["vgg16_ssd_voc_20", "vgg_ssd"]

from deeplite_torch_zoo.src.objectdetection.mb_ssd.repo.vision.ssd.vgg_ssd import create_vgg_ssd
from deeplite_torch_zoo.src.objectdetection.mb_ssd.config.vgg_ssd_config import VGG_CONFIG


model_urls = {
    "vgg16_ssd": "http://download.deeplite.ai/zoo/models/vgg16-ssd-voc-mp-0_7726-b1264e8beec69cbc.pth",
}


def vgg_ssd(num_classes=20, pretrained=False, progress=True, device='cuda'):
    model = create_vgg_ssd(num_classes + 1)
    config = VGG_CONFIG()
    model.config = config
    model.priors = config.priors.to(device)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["vgg16_ssd"], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)

    return model


def vgg16_ssd_voc_20(pretrained=False, progress=True, device='cuda'):
    return vgg_ssd(pretrained=pretrained, progress=progress, device=device)
