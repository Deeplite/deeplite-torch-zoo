
from pathlib import Path
import urllib.parse as urlparse
from collections import namedtuple

import deeplite_torch_zoo
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import YoloV5_6
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights


def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]


__all__ = [
    "yolo3",
    "YOLOV3_MODELS",
]

CFG_PATH = "deeplite_torch_zoo/src/objectdetection/configs/model_configs"
CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models"

model_urls = {
    "yolov3_voc_20": "yolo3-voc-0_839-a6149826183808aa.pth",
    "yolov3_voc_1": "yolov3-voc-1cls-0_888-1c73632fc187ef0c.pth",  # person
    "yolov3_voc_2": "yolov3-voc-2cls-0_911-b308f8a2686c19a6.pth",  # person and car
    "yolov3_lisa_11": "yolov3-lisa_11_830-663a0ec046402856.pth",
}

yolov3_cfg = {
    "yolov3": "yolov3.yaml",
    "yolov3_tiny": "yolov3-tiny.yaml",
    "yolov3_spp": "yolov3-spp.yaml",
}

YOLOV3_MODELS = list(yolov3_cfg.keys())


def yolo3(
    net="yolov3", _set_classes="voc_20", num_classes=20, pretrained=False,
    progress=True, device="cuda", **kwargs
):
    config_path = get_project_root() / CFG_PATH / yolov3_cfg[net]
    model = YoloV5_6(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"yolov3_{_set_classes}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)

    return model.to(device)


def make_wrapper_func(wrapper_name, net, _set_classes, num_classes):
    def wrapper_func(pretrained=False, progress=True, device="cuda"):
        return yolo3(
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
    'voc_20': ModelSet(20, ['yolov3']),
    'voc_1': ModelSet(1, ['yolov3']),
    'voc_2': ModelSet(2, ['yolov3']),
    'lisa_11': ModelSet(11, ['yolov3']),
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag.replace('v', ''), dataset]) # workaround for 'yolo3' -> 'yolov3' names
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
