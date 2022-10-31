import re
import urllib.parse as urlparse
from pathlib import Path

import deeplite_torch_zoo
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import \
    YoloV5_6
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
    model_name="yolo4s", dataset_name="voc", num_classes=20, pretrained=False,
    progress=True, device="cuda", ch=3,
):
    config_path = get_project_root() / CFG_PATH / yolov4_cfg[model_name]
    model = YoloV5_6(config_path, ch=ch, nc=num_classes)
    if pretrained:
        if f"{model_name}_{dataset_name}" not in model_urls:
            raise ValueError(f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}')
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{model_name}_{dataset_name}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


MODEL_TAG_TO_WRAPPER_FN_MAP = {
    "^yolo4[smlx]$": yolo4,
    "^yolo4[smlx]_leaky$": yolo4,
}


def make_wrapper_func(wrapper_name, model_name, dataset_name, num_classes):
    model_wrapper_fn = None
    for net_name, model_fn in MODEL_TAG_TO_WRAPPER_FN_MAP.items():
        if re.match(net_name, model_name):
            model_wrapper_fn = model_fn
    if model_wrapper_fn is None:
        raise ValueError(f'Could not find a wrapper function for model name {model_name}')

    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection')
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device="cuda"):
        return model_wrapper_fn(
            model_name=model_name,
            dataset_name=dataset_name,
            num_classes=num_classes,
            pretrained=pretrained,
            progress=progress,
            device=device,
        )
    wrapper_func.__name__ = wrapper_name
    return wrapper_func


model_list = list(yolov4_cfg.keys())
datasets = [('person_detection', 1), ('voc', 20), ('coco', 80), ('voc07', 20)]

for dataset_tag, n_classes in datasets:
    for model_tag in model_list:
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(name, model_tag, dataset_tag, n_classes)
        __all__.append(name)
