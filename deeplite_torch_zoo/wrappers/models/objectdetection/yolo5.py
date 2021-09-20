import torch
from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5 import YoloV5

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


__all__ = [
    "yolo5",
    "yolo5s_voc_20",
    "yolo5m_voc_20",
    "yolo5l_voc_20",
    "yolo5x_voc_20",
    "yolo5s_coco_80",
    "yolo5m_coco_80",
    "yolo5l_coco_80",
    "yolo5x_coco_80",
]

model_urls = {
    "yolov5s_voc_20": "http://download.deeplite.ai/zoo/models/yolo5s-voc-20classes_817-0325eb3aa0a02a50.pt",
    "yolov5m_voc_20": "http://download.deeplite.ai/zoo/models/yolo5m-voc-20classes_882-c2ad7d49ecfb27b3.pt",
    "yolov5l_voc_20": "http://download.deeplite.ai/zoo/models/yolo5l-voc-20classes_899-411aefb761eafaa3.pt",
    "yolov5x_voc_20": "http://download.deeplite.ai/zoo/models/yolo5x-voc-20classes_905-e8ddd018ae29751f.pt",
    "yolov5s_coco_80": "deeplite_torch_zoo/weight/yolo5s-coco-80classes.pt",
    "yolov5m_coco_80": "deeplite_torch_zoo/weight/yolo5m-coco-80classes.pt",
    "yolov5l_coco_80": "deeplite_torch_zoo/weight/yolo5l-coco-80classes.pt",
    "yolov5x_coco_80": "deeplite_torch_zoo/weight/yolo5x-coco-80classes.pt",
}

yolov5_cfg = {
    "yolov5s": "deeplite_torch_zoo/src/objectdetection/configs/yolov5s.yaml",
    "yolov5m": "deeplite_torch_zoo/src/objectdetection/configs/yolov5m.yaml",
    "yolov5l": "deeplite_torch_zoo/src/objectdetection/configs/yolov5l.yaml",
    "yolov5x": "deeplite_torch_zoo/src/objectdetection/configs/yolov5x.yaml",
}


def yolo5_local(
    net, pretrained=False, progress=True, num_classes=80, device="cuda", exclude=[]
):
    config_path = get_project_root() / yolov5_cfg[net]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        pretrained_model = torch.load(model_urls[net], map_location=device)
        pretrained_model.model[-1] = None
        pretrained_dict = pretrained_model.state_dict()
        model.load_state_dict(pretrained_dict, strict=False)
    return model.to(device)


def yolo5(
    net="yolov5s",
    pretrained=False,
    progress=True,
    num_classes=80,
    device="cuda",
    exclude=[],
):
    config_path = get_project_root() / yolov5_cfg[net]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        pretrained_model = torch.hub.load("ultralytics/yolov5", net, pretrained=True)
        pretrained_model.model[-1] = None
        pretrained_dict = pretrained_model.state_dict()
        model.load_state_dict(pretrained_dict, strict=False)
    return model.to(device)

def yolo5_voc(
    net="yolov5s", _set_classes="voc_20", pretrained=False, progress=True, device="cuda"
):
    config_path = get_project_root() / yolov5_cfg[net]
    model = YoloV5(config_path, ch=3, nc=20)
    if pretrained:
        pretrained_model = load_state_dict_from_url(
            model_urls[
                f"{net}_{_set_classes}"
            ],
            progress=progress,
            check_hash=True,
            map_location=device,
        )
        model.load_state_dict(pretrained_model)
    return model.to(device)


def yolo5s_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5_voc(
        net="yolov5s",
        _set_classes="voc_20",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5m_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5_voc(
        net="yolov5m",
        _set_classes="voc_20",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5l_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5_voc(
        net="yolov5l",
        _set_classes="voc_20",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5x_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5_voc(
        net="yolov5x",
        _set_classes="voc_20",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5_coco(
    net="yolov5s",
    _set_classes="coco_80",
    pretrained=False,
    progress=True,
    device="cuda",
):
    config_path = get_project_root() / yolov5_cfg[net]

    model = YoloV5(config_path, ch=3, nc=80)
    if pretrained:
        model = torch.load(
            model_urls[
                f"{net}_{_set_classes}"
            ],
            map_location=device,
        )
    return model.to(device)


def yolo5s_coco_80(pretrained=False, progress=True, device="cuda"):
    return yolo5_coco(
        net="yolov5s",
        _set_classes="coco_80",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5m_coco_80(pretrained=False, progress=True, device="cuda"):
    return yolo5_coco(
        net="yolov5m",
        _set_classes="coco_80",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5l_coco_80(pretrained=False, progress=True, device="cuda"):
    return yolo5_coco(
        net="yolov5l",
        _set_classes="coco_80",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5x_coco_80(pretrained=False, progress=True, device="cuda"):
    return yolo5_coco(
        net="yolov5x",
        _set_classes="coco_80",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
