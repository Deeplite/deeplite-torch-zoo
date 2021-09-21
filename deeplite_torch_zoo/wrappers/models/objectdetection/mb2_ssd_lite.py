from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from deeplite_torch_zoo.src.objectdetection.ssd.config.mobilenetv1_ssd_config import MOBILENET_CONFIG


__all__ = [
    "mb2_ssd_lite",
    "mb2_ssd_lite_subcls",
    "mb2_ssd_lite_voc_1",
    "mb2_ssd_lite_voc_2",
    "mb2_ssd_lite_voc_20",
]


model_urls = {
    "mb2_ssd_lite_voc_01": "http://download.deeplite.ai/zoo/models/mb2-ssd-lite-voc_1cls-0_664-8e73a022290eaae0.pth",
    "mb2_ssd_lite_voc_02": "http://download.deeplite.ai/zoo/models/mb2-ssd-lite-voc_2cls-0_716-38c909c73ccc5666.pth",
    "mb2_ssd_lite_voc_20": "http://download.deeplite.ai/zoo/models/mb2-ssd-lite-voc-mp-0_686-b0d1ac2c.pth",
}


def mb2_ssd_lite(
    net="mb2_ssd_lite",
    _dataset="voc_20",
    num_classes=20,
    pretrained=False,
    progress=True,
    device="cuda",
):
    model = create_mobilenetv2_ssd_lite(num_classes + 1)  # + 1 for background class
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


def mb2_ssd_lite_voc_20(pretrained=False, progress=True, device="cuda"):
    return mb2_ssd_lite(
        net="mb2_ssd_lite",
        _dataset="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def mb2_ssd_lite_voc_1(pretrained=False, progress=True, device="cuda"):
    return mb2_ssd_lite(
        net="mb2_ssd_lite",
        _dataset="voc_01",
        num_classes=1,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def mb2_ssd_lite_voc_2(pretrained=False, progress=True, device="cuda"):
    return mb2_ssd_lite(
        net="mb2_ssd_lite",
        _dataset="voc_02",
        num_classes=2,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def mb2_ssd_lite_subcls(num_classes=21, pretrained=True, progress=False, device="cuda"):
    model = create_mobilenetv2_ssd_lite(num_classes)
    if pretrained:
        pretrained_model = mb2_ssd_lite(pretrained=pretrained, progress=progress)
        state_dict = pretrained_model.state_dict()
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not (
                k.startswith("classification_headers")
                or k.startswith("regression_headers")
            )
        }
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)
