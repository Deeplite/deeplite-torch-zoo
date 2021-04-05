from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.objectdetection.ssd300.model.ssd300 import SSD300

__all__ = [
    "ssd300_resnet18_voc_20",
    "ssd300_resnet34_voc_20",
    "ssd300_resnet50_voc_20",
    "ssd300_vgg16_voc_20",
]
model_urls = {
    "ssd300_resnet18_voc_20": "http://download.deeplite.ai/zoo/models/ssd300-resnet18-voc20classes_580-cfc94e5b701953ba.pth",
    "ssd300_resnet34_voc_20": "http://download.deeplite.ai/zoo/models/ssd300-resnet34-voc20classes_654-eafd64758f6bfd1d.pth",
    "ssd300_resnet50_voc_20": "http://download.deeplite.ai/zoo/models/ssd300-resnet50-voc20classes_659-07069cb099a9a8b8.pth",
    "ssd300_vgg16_voc_20": "http://download.deeplite.ai/zoo/models/ssd300-vgg16-voc20classes_641-07cc9e5fecdcecc1.pth",
}


def ssd300(
    backbone="resnet50",
    data_set="voc_20",
    pretrained=False,
    progress=True,
    num_classes=20,
    device="cuda",
):
    model = SSD300(backbone=backbone, num_classes=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[
                "ssd300_{backbone}_{data_set}".format(
                    backbone=backbone, data_set=data_set
                )
            ],
            progress=progress,
            check_hash=True,
            map_location=device,
        )
        model.load_state_dict(state_dict, strict=False)

    return model.to(device)


def ssd300_resnet18_voc_20(pretrained=False, progress=False, device='cuda'):
    return ssd300(
        backbone="resnet18", data_set="voc_20", pretrained=pretrained, progress=progress, device=device
    )


def ssd300_resnet34_voc_20(pretrained=False, progress=False, device='cuda'):
    return ssd300(
        backbone="resnet34", data_set="voc_20", pretrained=pretrained, progress=progress, device=device
    )


def ssd300_resnet50_voc_20(pretrained=False, progress=False, device='cuda'):
    return ssd300(
        backbone="resnet50", data_set="voc_20", pretrained=pretrained, progress=progress, device=device
    )


def ssd300_vgg16_voc_20(pretrained=False, progress=False, device='cuda'):
    return ssd300(
        backbone="vgg16", data_set="voc_20", pretrained=pretrained, progress=progress, device=device
    )
