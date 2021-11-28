from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5 import YoloV5
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import YoloV5_6

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


__all__ = [
    "yolo5_local",
    "yolo5_6",
    "yolo5s_voc_20",
    "yolo5m_voc_20",
    "yolo5l_voc_20",
    "yolo5x_voc_20",
    "yolo5_6s_voc_20",
    "yolo5_6m_voc_20",
    "yolo5_6n_voc_20",
    "yolo5s_coco_80",
    "yolo5m_coco_80",
    "yolo5l_coco_80",
    "yolo5x_coco_80",
    "yolo5m_wider_face_8",
    "yolo5l_wider_face_8",
    "yolo5m_voc_24",
    "yolo5l_voc_24",
    "YOLOV5_MODELS",
]

model_urls = {
    "yolov5s_voc_20": "http://download.deeplite.ai/zoo/models/yolov5s_voc-0_837-1e922891b803a8b7.pt",
    "yolov5m_voc_20": "http://download.deeplite.ai/zoo/models/yolo5m-voc-20classes_882-1d8265513714a3f6.pt",
    "yolov5l_voc_20": "http://download.deeplite.ai/zoo/models/yolo5l-voc-20classes_899-411aefb761eafaa3.pt",
    "yolov5x_voc_20": "http://download.deeplite.ai/zoo/models/yolo5x-voc-20classes_905-e8ddd018ae29751f.pt",
    "yolov5s_coco_80": "deeplite_torch_zoo/weight/yolo5s-coco-80classes.pt",
    "yolov5m_coco_80": "deeplite_torch_zoo/weight/yolo5m-coco-80classes.pt",
    "yolov5l_coco_80": "deeplite_torch_zoo/weight/yolo5l-coco-80classes.pt",
    "yolov5x_coco_80": "deeplite_torch_zoo/weight/yolo5x-coco-80classes.pt",
    "yolov5l_wider_face_8": "http://download.deeplite.ai/zoo/models/yolo5l-widerface-8cls-898_cdedd11381dbf565.pt",
    "yolov5m_wider_face_8": "http://download.deeplite.ai/zoo/models/yolo5m-widerface-8cls-878_8a99aaf8b8b9157b.pt",
    "yolov5l_voc_24": "http://download.deeplite.ai/zoo/models/yolo5l_voc-24_885_391dfc95d193faf5.pt",
    "yolov5m_voc_24": "http://download.deeplite.ai/zoo/models/yolo5m_voc_24_871_54be57d3f5a35a7b.pt",
    "yolov5_6s_voc_20": "http://download.deeplite.ai/zoo/models/yolo5_6s-voc-20classes_821-6654b7ae075935fd.pt",
    "yolov5_6n_voc_20": "http://download.deeplite.ai/zoo/models/yolo5_6n-voc-20classes_762-a6b8573a32ebb4c8.pt",
    "yolov5_6m_voc_20": "http://download.deeplite.ai/zoo/models/yolo5_6s-voc-20classes_821-6654b7ae075935fd.pt",
}

yolov5_cfg = {
    "yolov5s": "deeplite_torch_zoo/src/objectdetection/configs/yolov5s.yaml",
    "yolov5m": "deeplite_torch_zoo/src/objectdetection/configs/yolov5m.yaml",
    "yolov5l": "deeplite_torch_zoo/src/objectdetection/configs/yolov5l.yaml",
    "yolov5x": "deeplite_torch_zoo/src/objectdetection/configs/yolov5x.yaml",
    "yolov5_6s": "deeplite_torch_zoo/src/objectdetection/configs/yolov5_6s.yaml",
    "yolov5_6m": "deeplite_torch_zoo/src/objectdetection/configs/yolov5_6m.yaml",
    "yolov5_6l": "deeplite_torch_zoo/src/objectdetection/configs/yolov5_6l.yaml",
    "yolov5_6x": "deeplite_torch_zoo/src/objectdetection/configs/yolov5_6x.yaml",
    "yolov5_6n": "deeplite_torch_zoo/src/objectdetection/configs/yolov5_6n.yaml",
}

YOLOV5_MODELS = list(yolov5_cfg.keys())


def yolo5_local(
    net, pretrained=False, num_classes=80, device="cuda", exclude=[], **kwargs
):
    config_path = get_project_root() / yolov5_cfg[net]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[f"{net}_voc_20"], map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if "model.24" not in k}
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def yolo5_6(
    net, num_classes=80, device="cuda", pretrained=False, exclude=[], **kwargs
):
    config_path = get_project_root() / yolov5_cfg[net]
    model = YoloV5_6(config_path, ch=3, nc=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[f"{net}_voc_20"], map_location=device)
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def yolo5(
    net="yolov5s", _set_classes="voc_20", num_classes=20, pretrained=False, progress=True, device="cuda"
):
    config_path = get_project_root() / yolov5_cfg[net]
    model = YoloV5(config_path, ch=3, nc=num_classes)
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
    return yolo5(
        net="yolov5s",
        _set_classes="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5m_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5m",
        _set_classes="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5m_voc_24(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5m",
        _set_classes="voc_24",
        num_classes=24,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5l_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5l",
        _set_classes="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5l_voc_24(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5l",
        _set_classes="voc_24",
        num_classes=24,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5x_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5x",
        _set_classes="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5_6s_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5_6(
        net="yolov5_6s",
        _set_classes="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5_6m_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5_6(
        net="yolov5_6m",
        _set_classes="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5_6n_voc_20(pretrained=False, progress=True, device="cuda"):
    return yolo5_6(
        net="yolov5_6n",
        _set_classes="voc_20",
        num_classes=20,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5s_coco_80(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5s",
        _set_classes="coco_80",
        num_classes=80,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5m_coco_80(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5m",
        _set_classes="coco_80",
        num_classes=80,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5l_coco_80(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5l",
        _set_classes="coco_80",
        num_classes=80,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5x_coco_80(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5x",
        _set_classes="coco_80",
        num_classes=80,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5l_wider_face_8(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5l",
        _set_classes="wider_face_8",
        num_classes=8,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo5m_wider_face_8(pretrained=False, progress=True, device="cuda"):
    return yolo5(
        net="yolov5m",
        _set_classes="wider_face_8",
        num_classes=8,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
