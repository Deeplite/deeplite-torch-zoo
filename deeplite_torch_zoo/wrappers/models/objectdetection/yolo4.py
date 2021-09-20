from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5 import YoloV5

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


# Uncomment the following imports for training purposes
# import sys
# sys.path.append("deeplite_torch_zoo/objectdetection/yolov4")
# from mish_cuda import *
# from models import *
__all__ = [
    "yolo4",
    "yolo4s_voc_20",
    "yolo4m_voc_20",
    "yolo4l_voc_20",
    "yolo4x_voc_20",
    "yolo4_lisa",
    "yolo4m_lisa_11",
    "yolo4l_leaky_voc_20"
]

model_urls = {
    "yolov4s_voc_20": "http://download.deeplite.ai/zoo/models/yolo4s-voc-20classes_850-270ddc5d43290a95.pth",
    "yolov4m_voc_20": "http://download.deeplite.ai/zoo/models/yolo4m-voc-20classes_885-b854caad9ca7fb7c.pth",
    "yolov4l_voc_20": "http://download.deeplite.ai/zoo/models/yolo4l-voc-20classes_872-9f54132ce2934fbf.pth",
    "yolov4x_voc_20": "http://download.deeplite.ai/zoo/models/yolo4x-voc-20classes_882-187f352b9d0d29c6.pth",
    "yolov4m_lisa_11": "http://download.deeplite.ai/zoo/models/yolov4m-lisa_11_880-6615c5e27557fab0.pth",
    "yolov4l_leaky_voc_20": "http://download.deeplite.ai/zoo/models/yolo4l-leaky-voc-20classes_891-2c0f78ee3938ade3.pt"
}

yolov4_cfg = {
    "yolov4s": "deeplite_torch_zoo/src/objectdetection/configs/yolov4s-mish.yaml",
    "yolov4m": "deeplite_torch_zoo/src/objectdetection/configs/yolov4m-mish.yaml",
    "yolov4l": "deeplite_torch_zoo/src/objectdetection/configs/yolov4l-mish.yaml",
    "yolov4x": "deeplite_torch_zoo/src/objectdetection/configs/yolov4x-mish.yaml",
    "yolov4l_leaky": "deeplite_torch_zoo/src/objectdetection/configs/yolov4l-leaky.yaml",
    "yolov5s": "deeplite_torch_zoo/src/objectdetection/configs/yolov5s.yaml",
    "yolov5m": "deeplite_torch_zoo/src/objectdetection/configs/yolov5m.yaml",
    "yolov5l": "deeplite_torch_zoo/src/objectdetection/configs/yolov5l.yaml",
    "yolov5x": "deeplite_torch_zoo/src/objectdetection/configs/yolov5x.yaml",
}



def _yolo4(
    net="yolov4s", _set_classes="voc_20", num_classes=20, pretrained=False, progress=True, device="cuda"
):
    config_path = get_project_root() / yolov4_cfg[net]
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

def yolo4(
    net="yolov4s", pretrained=False, progress=True, num_classes=80, device="cuda"
):
    config_path = get_project_root() / yolov4_cfg[net]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        pretrained_model = _yolo4(net=net, _set_classes="voc_20", pretrained=True)
        pretrained_model.model[-1] = None
        pretrained_dict = pretrained_model.state_dict()
        model.load_state_dict(pretrained_dict, strict=False)
    return model.to(device)



def yolo4_lisa(
    net="yolov4s", pretrained=False, progress=True, num_classes=80, device="cuda"
):
    config_path = get_project_root() / yolov4_cfg[net]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        pretrained_model = yolo4m_voc_20(pretrained=True)
        pretrained_model.model[-1] = None
        pretrained_dict = pretrained_model.state_dict()
        model.load_state_dict(pretrained_dict, strict=False)
    return model.to(device)


def yolo4s_voc_20(pretrained=False, progress=True, device="cuda"):
    return _yolo4(
        net="yolov4s",
        _set_classes="voc_20",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo4m_voc_20(pretrained=False, progress=True, device="cuda"):
    return _yolo4(
        net="yolov4m",
        _set_classes="voc_20",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo4m_lisa_11(pretrained=False, progress=True, device="cuda"):
    return _yolo4(
        net="yolov4m",
        _set_classes="lisa_11",
        num_classes=11,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo4l_voc_20(pretrained=False, progress=True, device="cuda"):
    return _yolo4(
        net="yolov4l",
        _set_classes="voc_20",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo4x_voc_20(pretrained=False, progress=True, device="cuda"):
    return _yolo4(
        net="yolov4x",
        _set_classes="voc_20",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def yolo4l_leaky_voc_20(pretrained=False, progress=True, device="cuda"):
    return _yolo4(
        net="yolov4l_leaky",
        _set_classes="voc_20",
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
