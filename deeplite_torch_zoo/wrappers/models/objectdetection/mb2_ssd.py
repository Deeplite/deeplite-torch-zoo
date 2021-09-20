
from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.objectdetection.ssd.models.mobilenetv2_ssd import create_mobilenetv2_ssd
from deeplite_torch_zoo.src.objectdetection.ssd.config.mobilenetv1_ssd_config import MOBILENET_CONFIG

__all__ = [
    "mb2_ssd",
    "mb2_ssd_coco_gm_6",
    "mb2_ssd_voc_20",
    "mb2_ssd_coco_80",
]

model_urls = {
    "mb2_ssd_coco_80": "http://download.deeplite.ai/zoo/models/mb2-ssd-coco-80-0_303-78c63d5b07edaeaf.pth",
    "mb2_ssd_coco_gm_6": "http://download.deeplite.ai/zoo/models/mb2-ssd-coco-6-0_455-1ce14b4728801c61.pth",
    "mb2_ssd_voc_20": "http://download.deeplite.ai/zoo/models/mb2-ssd-voc-20-0_440-8402e1cfc3e076b1.pth",
}


def mb2_ssd(
    net="mb2_ssd",
    _dataset="voc_20",
    num_classes=20,
    pretrained=False,
    progress=True,
    device="cuda",
):
    model = create_mobilenetv2_ssd(num_classes + 1)  # + 1 for background class
    config = MOBILENET_CONFIG()
    model.config = config
    model.priors = config.priors.to(device)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[f"{net}_{_dataset}"],
            progress=progress,
            check_hash=True,
            map_location=device,
        )
        model.load_state_dict(state_dict)

    return model.to(device)


def mb2_ssd_coco_gm_6(pretrained=False, progress=True, device="cuda"):
    return mb2_ssd(
        net="mb2_ssd",
        _dataset="coco_gm_6",
        num_classes=6,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def mb2_ssd_voc_20(pretrained=False, progress=True, device="cuda"):
    return mb2_ssd(
        net="mb2_ssd",
        _dataset="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def mb2_ssd_coco_80(pretrained=False, progress=True, device="cuda"):
    return mb2_ssd(
        net="mb2_ssd",
        _dataset="coco_80",
        num_classes=80,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
