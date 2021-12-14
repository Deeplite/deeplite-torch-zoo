from pathlib import Path
from collections import namedtuple

from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5 import YoloV5
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


# Uncomment the following imports for training purposes
# import sys
# sys.path.append("deeplite_torch_zoo/objectdetection/yolov4")
# from mish_cuda import *
# from models import *

__all__ = [
    "yolo4",
    "YOLOV4_MODELS",
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
    "yolov4s": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4s-mish.yaml",
    "yolov4m": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4m-mish.yaml",
    "yolov4l": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4l-mish.yaml",
    "yolov4x": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4x-mish.yaml",
    "yolov4l_leaky": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4l-leaky.yaml",
    "yolov5s": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5s.yaml",
    "yolov5m": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5m.yaml",
    "yolov5l": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5l.yaml",
    "yolov5x": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5x.yaml",
}

YOLOV4_MODELS = list(yolov4_cfg.keys())


def yolo4(
    net="yolov4s", _set_classes="voc_20", num_classes=20, pretrained=False,
    progress=True, device="cuda",
):
    config_path = get_project_root() / yolov4_cfg[net]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = model_urls[f"{net}_{_set_classes}"]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def make_wrapper_func(wrapper_name, net, _set_classes, num_classes):
    def wrapper_func(pretrained=False, progress=True, device="cuda"):
        return yolo4(
            net=net,
            _set_classes=_set_classes,
            num_classes=num_classes,
            pretrained=pretrained,
            progress=progress,
            device=device,
        )
    wrapper_func.__name__ = wrapper_name
    return wrapper_func


ModelSet = namedtuple('ModelSet', ['num_classes', 'model_list'])
wrapper_funcs = {
    'voc_20': ModelSet(20, ['yolov4s', 'yolov4m', 'yolov4l', 'yolov4x', 'yolov4l_leaky']),
    'lisa_11': ModelSet(11, ['yolov4m'])
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag.replace('v', ''), dataset]) # workaround for 'yolo4' -> 'yolov4' names
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
