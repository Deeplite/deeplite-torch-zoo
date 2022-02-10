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


def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]


__all__ = [
    "yolo5",
    "yolo5_6",
    "YOLOV5_MODELS",
]

CFG_PATH = "deeplite_torch_zoo/src/objectdetection/configs/model_configs"
CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/"

model_urls = {
    "yolov5s_voc_20": "yolov5s_voc-0_837-1e922891b803a8b7.pt",
    "yolov5m_voc_20": "yolo5m-voc-20classes_882-1d8265513714a3f6.pt",
    "yolov5l_voc_20": "yolo5l-voc-20classes_899-411aefb761eafaa3.pt",
    "yolov5x_voc_20": "yolo5x-voc-20classes_905-e8ddd018ae29751f.pt",
    "yolov5l_wider_face_8": "yolo5l-widerface-8cls-898_cdedd11381dbf565.pt",
    "yolov5m_wider_face_8": "yolo5m-widerface-8cls-878_8a99aaf8b8b9157b.pt",
    "yolov5l_voc_24": "yolo5l_voc-24_885_391dfc95d193faf5.pt",
    "yolov5m_voc_24": "yolo5m_voc_24_871_54be57d3f5a35a7b.pt",
    "yolov5_6s_voc_20": "yolo5_6s-voc-20classes_871-4ceb1b22b227c05c.pt",
    "yolov5_6n_voc_20": "yolo5_6n-voc-20classes_762-a6b8573a32ebb4c8.pt",
    "yolov5_6m_voc_20": "yolo5_6m-voc-20classes_902-50c151baffbf896e.pt",
    "yolov5_6s_relu_voc_20": "yolov5_6s_relu-voc-20classes-819_a35dff53b174e383.pt",
    "yolov5_6m_relu_voc_20": "yolov5_6m_relu-voc-20classes-856_c5c23135e6d5012f.pt",
    "yolov5_6s_coco_80": "yolov5_6s-coco-80classes_301-8ff1dabeec225366.pt",
    "yolov5_6m_coco_80": "yolov5_6m-coco-80classes_374-f93fa94b629c45ab.pt",
    "yolov5_6n_coco_80": "yolov5_6n-coco-80classes_211-e9e44a7de1f08ea2.pt",
    "yolov5_6s_person_detection_1": "yolov5_6s-person-detection-1class_738-9e9ac9dae14b0dcd.pt",
    "yolov5_6n_person_detection_1": "yolov5_6n-person-detection-1class_696-fff2a2c720e20752.pt",
    "yolov5_6s_relu_person_detection_1": "yolov5_6s_relu-person-detection-1class_682-45ae979a06b80767.pt",
    "yolov5_6m_relu_person_detection_1": "yolov5_6m_relu-person-detection-1class_709-3f59321c540d2d1c.pt",
    "yolov5_6n_relu_person_detection_1": "yolov5_6n_relu-person-detection-1class_621-6794298f12d33ba8.pt",
    "yolov5_6n_voc07_20": "yolov5_6n-voc07-20classes-620_037230667eff7b12.pt",
    "yolov5_6s_voc07_20": "yolov5_6s-voc07-20classes-687_4d221fd4edc09ce1.pt",
    "yolov5_6s_relu_voc_20": "yolov5_6s_relu-voc-20classes-819_a35dff53b174e383.pt",
    "yolov5_6m_relu_voc_20": "yolov5_6m_relu-voc-20classes-856_c5c23135e6d5012f.pt",
}

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
}

YOLOV5_MODELS = []
MODEL_NAME_SUFFICES = ('relu', )
for model_name_tag in yolov5_cfg:
    YOLOV5_MODELS.append(model_name_tag)
    for model_name_suffix in MODEL_NAME_SUFFICES:
        YOLOV5_MODELS.append('_'.join((model_name_tag, model_name_suffix)))


@MODEL_WRAPPER_REGISTRY.register('yolo5')
def yolo5(
    net="yolov5s", _set_classes="voc_20", num_classes=20,
    pretrained=False, progress=True, device="cuda"
):
    config_key = net
    config_path = get_project_root() / CFG_PATH / yolov5_cfg[config_key]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{net}_{_set_classes}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register('yolo5_6')
def yolo5_6(
    net="yolov5_6s", _set_classes="voc_20", num_classes=20, activation_type=None,
    pretrained=False, progress=True, device="cuda"
):
    for suffix in MODEL_NAME_SUFFICES:
        config_key = re.sub(f'\_{suffix}$', '', net) # pylint: disable=W1401
    config_path = get_project_root() / CFG_PATH / yolov5_cfg[config_key]
    model = YoloV5_6(config_path, ch=3, nc=num_classes, activation_type=activation_type)
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{net}_{_set_classes}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


MODEL_TAG_TO_WRAPPER_FN_MAP = {
    "^yolov5[smlx]$": yolo5,
    "^yolov5_6[nsmlx]$": yolo5_6,
    "^yolov5_6[nsmlx]_relu$": partial(yolo5_6, activation_type="relu"),
}

def make_wrapper_func(wrapper_name, net, _set_classes, num_classes):

    for net_name, model_fn in MODEL_TAG_TO_WRAPPER_FN_MAP.items():
        if re.match(net_name, net):
            model_wrapper_fn = model_fn

    model_name = net.replace('v','')
    @MODEL_WRAPPER_REGISTRY.register(model_name, _set_classes)
    def wrapper_func(pretrained=False, progress=True, device="cuda"):
        return model_wrapper_fn(
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
    'person_detection_1': ModelSet(1, ['yolov5_6n', 'yolov5_6s',
        'yolov5_6n_relu', 'yolov5_6s_relu', 'yolov5_6m_relu']),
    'voc_20': ModelSet(20, ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
        'yolov5_6n', 'yolov5_6s', 'yolov5_6m', 'yolov5_6m_relu', 'yolov5_6s_relu']),
    'voc_24': ModelSet(24, ['yolov5m', 'yolov5l']),
    'wider_face_8': ModelSet(8, ['yolov5m', 'yolov5l']),
    'coco_80': ModelSet(80, ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
        'yolov5_6n', 'yolov5_6s', 'yolov5_6m']),
    'voc07_20': ModelSet(20, ['yolov5_6n', 'yolov5_6s']),
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag.replace('v', ''), dataset]) # workaround for 'yolo5' -> 'yolov5' names
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
