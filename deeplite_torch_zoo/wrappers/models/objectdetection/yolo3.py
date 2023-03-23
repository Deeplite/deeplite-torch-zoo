import re
import urllib.parse as urlparse
from functools import partial
from pathlib import Path

import deeplite_torch_zoo
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import \
    YOLOModel
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

__all__ = []


def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]

CFG_PATH = "deeplite_torch_zoo/src/objectdetection/yolov5/configs/model_configs/yolo3"
CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/"

model_urls = {
    "yolo3_voc": "yolov3-voc-20classes-912_c94dc14873207830.pt",
}

yolov3_cfg = {
    "yolo3": "yolov3.yaml",
    "yolo3n": "yolov3n.yaml",
    "yolo3s": "yolov3s.yaml",
    "yolo3m": "yolov3m.yaml",
    "yolo3l": "yolov3l.yaml",
    "yolo3_d1w5": "yolo3_d1w5.yaml",
    "yolo3_d1w25": "yolo3_d1w25.yaml",
    "yolo3_d1w75": "yolo3_d1w75.yaml",
    "yolo3_d33w1": "yolo3_d33w1.yaml",
    "yolo3_d33w75": "yolo3_d33w75.yaml",
    "yolo3_d67w1": "yolo3_d67w1.yaml",
    "yolo3_d67w5": "yolo3_d67w5.yaml",
    "yolo3_d67w25": "yolo3_d67w25.yaml",
    #######
    "yolo3_tiny": "yolov3-tiny.yaml",
    "yolo3_spp": "yolov3-spp.yaml",
}

MODEL_NAME_SUFFICES = ('relu', 'hswish')

def yolo3(
    model_name="yolo3", dataset_name="voc", num_classes=20, activation_type=None,
    depth_mul=None, width_mul=None, channel_divisor=8, pretrained=False, progress=True, device="cuda", ch=3,
):
    config_key = model_name
    for suffix in MODEL_NAME_SUFFICES:
        config_key = re.sub(f'\_{suffix}$', '', config_key) # pylint: disable=W1401
    config_path = get_project_root() / CFG_PATH / yolov3_cfg[config_key]
    model = YOLOModel(
        config_path,
        ch=ch,
        nc=num_classes,
        activation_type=activation_type,
        depth_mul=depth_mul,
        width_mul=width_mul,
        channel_divisor=channel_divisor,
    )
    if pretrained:
        if f"{model_name}_{dataset_name}" not in model_urls:
            raise ValueError(f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. \n'
                              'Use pretrained=False if you want to create a untrained model.')
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{model_name}_{dataset_name}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)

    return model.to(device)


MODEL_TAG_TO_WRAPPER_FN_MAP = {
    "yolo3": yolo3,
    "^yolo3[nsmlx]$": yolo3,
    "^yolo3[nsmlx]_relu$": partial(yolo3, activation_type="relu"),
    "^yolo3[nsmlx]_hswish$": partial(yolo3, activation_type="hardswish"),
    "yolo3_d*w*": yolo3,
}

def make_wrapper_func(wrapper_name, model_name, dataset_name, num_classes):

    model_wrapper_fn = None
    for net_name, model_fn in MODEL_TAG_TO_WRAPPER_FN_MAP.items():
        if re.match(net_name, model_name):
            model_wrapper_fn = model_fn

    has_checkpoint = True
    if f"{model_name}_{dataset_name}" not in model_urls:
        has_checkpoint = False
    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection', has_checkpoint=has_checkpoint)
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device="cuda", **kwargs):
        return model_wrapper_fn(
            model_name=model_name,
            dataset_name=dataset_name,
            num_classes=num_classes,
            pretrained=pretrained,
            progress=progress,
            device=device,
            **kwargs
        )
    wrapper_func.__name__ = wrapper_name
    return wrapper_func


model_list = list(yolov3_cfg.keys())
for model_name_suffix in MODEL_NAME_SUFFICES:
    model_list += [f'{model_name}_{model_name_suffix}' for model_name in yolov3_cfg]
datasets = [('person_detection', 1), ('voc', 20), ('coco', 80), ('voc07', 20)]

for dataset_tag, n_classes in datasets:
    for model_tag in model_list:
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(name, model_tag, dataset_tag, n_classes)
        __all__.append(name)
