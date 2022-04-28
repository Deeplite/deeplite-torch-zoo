import re
import urllib.parse as urlparse
from pathlib import Path
from collections import namedtuple

import deeplite_torch_zoo
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import YoloV5_6
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

__all__ = []


def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]


CFG_PATH = "deeplite_torch_zoo/src/objectdetection/yolov5/configs/model_configs"
CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/"

model_urls = {
    "yolo4s_voc": "yolov4s-voc-20classes_849_58041e8852a4b2e2.pt",
    "yolo4m_voc": "yolov4m-voc-20classes_874_e0c8e179992b5da2.pt",
    "yolo4l_voc": "yolo4l-voc-20classes_872-9f54132ce2934fbf.pth",
    "yolo4x_voc": "yolo4x-voc-20classes_882-187f352b9d0d29c6.pth",
    "yolo4l_leaky_voc": "yolo4l-leaky-voc-20classes_891-2c0f78ee3938ade3.pt",
    "yolo4s_coco": "yolov4_6s-coco-80classes-288_b112910223d6c56d.pt",
    "yolo4m_coco": "yolov4_6m-coco-80classes-309_02b2013002a4724b.pt",
}

yolov4_cfg = {
    "yolo4s": "yolov4s-mish.yaml",
    "yolo4m": "yolov4m-mish.yaml",
    "yolo4l": "yolov4l-mish.yaml",
    "yolo4x": "yolov4x-mish.yaml",
    "yolo4l_leaky": "yolov4l-leaky.yaml",
}


def yolo4(
    net="yolo4s", dataset_name="voc", num_classes=20, pretrained=False,
    progress=True, device="cuda",
):
    config_path = get_project_root() / CFG_PATH / yolov4_cfg[net]
    model = YoloV5_6(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{net}_{dataset_name}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


MODEL_TAG_TO_WRAPPER_FN_MAP = {
    "^yolo4[smlx]$": yolo4,
    "^yolo4[smlx]_leaky$": yolo4,
}


def make_wrapper_func(wrapper_name, model_name, dataset_name, num_classes):
    for net_name, model_fn in MODEL_TAG_TO_WRAPPER_FN_MAP.items():
        if re.match(net_name, model_name):
            model_wrapper_fn = model_fn

    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection')
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device="cuda"):
        return model_wrapper_fn(
            net=model_name,
            dataset_name=dataset_name,
            num_classes=num_classes,
            pretrained=pretrained,
            progress=progress,
            device=device,
        )
    wrapper_func.__name__ = wrapper_name
    return wrapper_func


ModelSet = namedtuple('ModelSet', ['num_classes', 'model_list'])
wrapper_funcs = {
    'voc': ModelSet(20, ['yolo4s', 'yolo4m', 'yolo4l', 'yolo4x', 'yolo4l_leaky']),
    'coco': ModelSet(80, ['yolo4s', 'yolo4m']),
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag, dataset])
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
