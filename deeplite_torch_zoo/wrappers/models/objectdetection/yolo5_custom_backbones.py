import re
import urllib.parse as urlparse
from pathlib import Path
from collections import namedtuple

import deeplite_torch_zoo
from deeplite_torch_zoo.src.objectdetection.yolov5_custom_backbone.model import YoloCustom
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = []


def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]


CFG_PATH = "deeplite_torch_zoo/src/objectdetection/yolov5_custom_backbone/configs/"
CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/"

model_urls = {
    "yolo5_vgg16_bn_voc_20": None,
}

yolo5_custom_cfg = {
    "yolo5_vgg16_bn": "yolov5_vgg.yaml",
}


def yolo5_custom(
    net="yolo5_vgg", dataset_name="voc_20", num_classes=20,
    pretrained=False, progress=True, device="cuda"
):
    config_key = net
    config_path = get_project_root() / CFG_PATH / yolo5_custom_cfg[config_key]
    model = YoloCustom(config_path, nc=num_classes)
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{net}_{dataset_name}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)

MODEL_TAG_TO_WRAPPER_FN_MAP = {
    "^yolo5_vgg16_bn$": yolo5_custom,
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
    'voc_20': ModelSet(20, ['yolo5_vgg16_bn']),
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag, dataset])
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
