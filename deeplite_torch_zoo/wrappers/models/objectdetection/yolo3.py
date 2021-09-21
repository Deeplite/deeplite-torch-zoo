
from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.objectdetection.yolov3.model.yolov3 import Yolov3

__all__ = [
    "yolo3",
    "yolo3_voc_1",
    "yolo3_voc_2",
    "yolo3_voc_6",
    "yolo3_voc_20",
    "yolo3_lisa_11",
]


model_urls = {
    "yolov3_voc_20": "http://download.deeplite.ai/zoo/models/yolo3-voc-0_839-a6149826183808aa.pth",
    "yolov3_voc_01": "http://download.deeplite.ai/zoo/models/yolov3-voc-1cls-0_888-1c73632fc187ef0c.pth",  # person
    "yolov3_voc_02": "http://download.deeplite.ai/zoo/models/yolov3-voc-2cls-0_911-b308f8a2686c19a6.pth",  # person and car
    "yolov3_voc_06": "http://download.deeplite.ai/zoo/models/yolov3-voc-6classes-0_904-14204b730c45d701.pth",
    "yolov3_lisa_11": "http://download.deeplite.ai/zoo/models/yolov3-lisa_11_830-663a0ec046402856.pth",
}


def yolo3(pretrained=False, progress=True, num_classes=20, device="cuda"):
    model = Yolov3(num_classes=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["yolov3_voc_20"],
            progress=progress,
            check_hash=True,
            map_location=device,
        )
        state_dict = {k: v for k, v in state_dict.items() if "fpn" not in k}
        model.load_state_dict(state_dict, strict=False)

    return model.to(device)


def yolo3_voc(
    _set_classes="voc_20", num_classes=20, pretrained=False, progress=True, device="cuda"
):
    model = Yolov3(num_classes=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[f"yolov3_{_set_classes}"],
            progress=progress,
            check_hash=True,
            map_location=device,
        )
        model.load_state_dict(state_dict)

    return model.to(device)


def yolo3_voc_1(pretrained=True, progress=False, device="cuda"):
    return yolo3_voc(
        _set_classes="voc_01",
        num_classes=1,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo3_voc_2(pretrained=True, progress=False, device="cuda"):
    return yolo3_voc(
        _set_classes="voc_02",
        num_classes=2,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo3_voc_6(pretrained=True, progress=False, device="cuda"):
    """
    Should be deprecated. The exact classes are lost
    """
    return yolo3_voc(
        _set_classes="voc_06",
        num_classes=6,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo3_voc_20(pretrained=True, progress=False, device="cuda"):
    return yolo3_voc(
        _set_classes="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo3_lisa_11(pretrained=True, progress=False, device="cuda"):
    return yolo3_voc(
        _set_classes="lisa_11",
        num_classes=11,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
