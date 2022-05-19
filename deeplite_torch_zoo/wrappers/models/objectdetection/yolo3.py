
from pathlib import Path
import urllib.parse as urlparse
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
    "yolo3_voc": "yolov3-voc-20classes-912_c94dc14873207830.pt",
}

yolov3_cfg = {
    "yolo3": "yolov3.yaml",
    "yolo3_tiny": "yolov3-tiny.yaml",
    "yolo3_spp": "yolov3-spp.yaml",
}


def yolo3(
    net="yolo3", dataset_name="voc", num_classes=20, pretrained=False,
    progress=True, device="cuda", **kwargs
):
    config_path = get_project_root() / CFG_PATH / yolov3_cfg[net]
    model = YoloV5_6(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{net}_{dataset_name}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)

    return model.to(device)


def make_wrapper_func(wrapper_name, model_name, dataset_name, num_classes):
    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection')
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device="cuda"):
        return yolo3(
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
    'voc': ModelSet(20, ['yolo3']),
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag, dataset])
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
