import re
import urllib.parse as urlparse
from pathlib import Path
from functools import partial
from collections import namedtuple

import deeplite_torch_zoo
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5 import YoloV5
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import YoloV5_6
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

__all__ = []


def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]


CFG_PATH = "deeplite_torch_zoo/src/objectdetection/yolov5/configs/model_configs"
CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/"

model_urls = {
    "yolov5_6s_coco_80": "yolov5_6s-coco-80classes_301-8ff1dabeec225366.pt",
    "yolov5_6m_coco_80": "yolov5_6m-coco-80classes_374-f93fa94b629c45ab.pt",
    "yolov5_6n_coco_80": "yolov5_6n-coco-80classes_211-e9e44a7de1f08ea2.pt",
    "yolov5_6sa_coco_80": "yolov5_6sa-coco-80classes_297-6c1972b5f7ae6ab6.pt",
    "yolov5_6ma_coco_80": "yolov5_6ma-coco-80classes_365-4756729c4f6a834f.pt",
    "yolov5_6n_hswish_coco_80": "yolov5_6n_hswish-coco-80classes-183-a2fed163ec98352a.pt",
    "yolov5_6n_relu_coco_80": "yolov5_6n_relu-coco-80classes-167-7b6609497c63df79.pt",
}

voc_model_urls = {
    "yolov5s_voc_20": "yolov5s_voc-0_837-1e922891b803a8b7.pt",
    "yolov5m_voc_20": "yolo5m-voc-20classes_882-1d8265513714a3f6.pt",
    "yolov5l_voc_20": "yolo5l-voc-20classes_899-411aefb761eafaa3.pt",
    "yolov5x_voc_20": "yolo5x-voc-20classes_905-e8ddd018ae29751f.pt",
    "yolov5_6n_voc_20": "yolo5_6n-voc-20classes_762-a6b8573a32ebb4c8.pt",
    "yolov5_6s_voc_20": "yolo5_6s-voc-20classes_871-4ceb1b22b227c05c.pt",
    "yolov5_6m_voc_20": "yolo5_6m-voc-20classes_902-50c151baffbf896e.pt",
    "yolov5_6l_voc_20": "yolov5_6l-voc-20classes_875_3fb90f0c405f170c.pt",
    "yolov5_6x_voc_20": "yolov5_6x-voc-20classes_884_a2b6fb7234218cf6.pt",
    "yolov5_6n_voc07_20": "yolov5_6n-voc07-20classes-620_037230667eff7b12.pt",
    "yolov5_6s_voc07_20": "yolov5_6s-voc07-20classes-687_4d221fd4edc09ce1.pt",
    "yolov5_6s_relu_voc_20": "yolov5_6s_relu-voc-20classes-819_a35dff53b174e383.pt",
    "yolov5_6m_relu_voc_20": "yolov5_6m_relu-voc-20classes-856_c5c23135e6d5012f.pt",
}

person_detection_model_urls = {
    "yolov5_6n_person_detection_1": "yolov5_6n-person-detection-1class_696-fff2a2c720e20752.pt",
    "yolov5_6s_person_detection_1": "yolov5_6s-person-detection-1class_738-9e9ac9dae14b0dcd.pt",
    "yolov5_6n_relu_person_detection_1": "yolov5_6n_relu-person-detection-1class_621-6794298f12d33ba8.pt",
    "yolov5_6s_relu_person_detection_1": "yolov5_6s_relu-person-detection-1class_682-45ae979a06b80767.pt",
    "yolov5_6m_relu_person_detection_1": "yolov5_6m_relu-person-detection-1class_709-3f59321c540d2d1c.pt",
    "yolov5_6sa_person_detection_1": "yolov5_6sa-person-detection-1class_659_015807ae6899af0f.pt",
}

model_urls.update(voc_model_urls)
model_urls.update(person_detection_model_urls)


yolov5_cfg = {
    "yolov5s": "yolov5s.yaml",
    "yolov5m": "yolov5m.yaml",
    "yolov5l": "yolov5l.yaml",
    "yolov5x": "yolov5x.yaml",
    "yolov5_6s": "yolov5_6s.yaml",
    "yolov5_6m": "yolov5_6m.yaml",
    "yolov5_6l": "yolov5_6l.yaml",
    "yolov5_6x": "yolov5_6x.yaml",
    "yolov5_6n": "yolov5_6n.yaml",
    "yolov5_6sa": "yolov5_6sa.yaml",
    "yolov5_6ma": "yolov5_6ma.yaml",
}


MODEL_NAME_SUFFICES = ('relu', 'hswish')


def yolo5(
    net="yolov5s", dataset_name="voc_20", num_classes=20,
    pretrained=False, progress=True, device="cuda"
):
    config_key = net
    config_path = get_project_root() / CFG_PATH / yolov5_cfg[config_key]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{net}_{dataset_name}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def yolo5_6(
    net="yolov5_6s", dataset_name="voc_20", num_classes=20, activation_type=None,
    pretrained=False, progress=True, device="cuda"
):
    config_key = net
    for suffix in MODEL_NAME_SUFFICES:
        config_key = re.sub(f'\_{suffix}$', '', config_key) # pylint: disable=W1401
    config_path = get_project_root() / CFG_PATH / yolov5_cfg[config_key]
    model = YoloV5_6(config_path, ch=3, nc=num_classes, activation_type=activation_type)
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{net}_{dataset_name}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


MODEL_TAG_TO_WRAPPER_FN_MAP = {
    "^yolov5[smlx]$": yolo5,
    "^yolov5_6[nsmlx]$": yolo5_6,
    "^yolov5_6[nsmlx]a$": yolo5_6,
    "^yolov5_6[nsmlx]_relu$": partial(yolo5_6, activation_type="relu"),
    "^yolov5_6[nsmlx]_hswish$": partial(yolo5_6, activation_type="hardswish"),
}

def make_wrapper_func(wrapper_name, net, dataset_name, num_classes):

    for net_name, model_fn in MODEL_TAG_TO_WRAPPER_FN_MAP.items():
        if re.match(net_name, net):
            model_wrapper_fn = model_fn

    model_name = net.replace('v', '')
    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection')
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device="cuda"):
        return model_wrapper_fn(
            net=net,
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
    'person_detection_1': ModelSet(1, ['yolov5_6n', 'yolov5_6s',
        'yolov5_6n_relu', 'yolov5_6s_relu', 'yolov5_6m_relu', 'yolov5_6sa']),
    'voc_20': ModelSet(20, ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
        'yolov5_6n', 'yolov5_6s', 'yolov5_6m', 'yolov5_6l', 'yolov5_6x',
        'yolov5_6m_relu', 'yolov5_6s_relu']),
    'coco_80': ModelSet(80, ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
        'yolov5_6n', 'yolov5_6s', 'yolov5_6m', 'yolov5_6sa', 'yolov5_6ma',
        'yolov5_6n_hswish', 'yolov5_6n_relu']),
    'voc07_20': ModelSet(20, ['yolov5_6n', 'yolov5_6s']),
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag.replace('v', ''), dataset]) # workaround for 'yolo5' -> 'yolov5' names
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
