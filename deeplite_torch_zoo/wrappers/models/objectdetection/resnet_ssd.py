from torch.hub import load_state_dict_from_url

__all__ = ["resnet18_ssd_voc_20", "resnet34_ssd_voc_20", "resnet50_ssd_voc_20"]

from deeplite_torch_zoo.src.objectdetection.ssd.models.resnet_ssd import create_resnet_ssd
from deeplite_torch_zoo.src.objectdetection.ssd.config.vgg_ssd_config import VGG_CONFIG as CONFIG


model_urls = {
    "resnet18_voc_20": "http://download.deeplite.ai/zoo/models/resnet18-ssd_voc_AP_0_728-564518d0c865972b.pth",
    "resnet34_voc_20": "http://download.deeplite.ai/zoo/models/resnet34_ssd-voc_760-a102a7ca6564ab44.pth",
    "resnet50_voc_20": "http://download.deeplite.ai/zoo/models/resnet50_ssd-voc_766-d934cbe063398fcd.pth",
}


def resnet_ssd(backbone="resnet18", dataset="voc_20", num_classes=20, pretrained=False, progress=True, device='cuda'):
    model = create_resnet_ssd(num_classes + 1, backbone=backbone)
    config = CONFIG()
    model.config = config
    model.priors = config.priors.to(device)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[f"{backbone}_{dataset}"], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)

    return model.to(device)


def resnet18_ssd_voc_20(pretrained=False, progress=True, device='cuda'):
    return resnet_ssd(backbone="resnet18", dataset="voc_20", num_classes=20, pretrained=pretrained, progress=progress, device=device)


def resnet34_ssd_voc_20(pretrained=False, progress=True, device='cuda'):
    return resnet_ssd(backbone="resnet34", dataset="voc_20", num_classes=20, pretrained=pretrained, progress=progress, device=device)


def resnet50_ssd_voc_20(pretrained=False, progress=True, device='cuda'):
    return resnet_ssd(backbone="resnet50", dataset="voc_20", num_classes=20, pretrained=pretrained, progress=progress, device=device)
