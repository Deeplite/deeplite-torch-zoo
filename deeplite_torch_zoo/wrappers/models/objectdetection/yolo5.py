import re
from pathlib import Path
from collections import namedtuple

from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5 import YoloV5
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import YoloV5_6
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


__all__ = [
    "yolo5",
    "yolo5_6",
    "YOLOV5_MODELS",
]

model_urls = {
    "yolov5s_voc_20": "http://download.deeplite.ai/zoo/models/yolov5s_voc-0_837-1e922891b803a8b7.pt",
    "yolov5m_voc_20": "http://download.deeplite.ai/zoo/models/yolo5m-voc-20classes_882-1d8265513714a3f6.pt",
    "yolov5l_voc_20": "http://download.deeplite.ai/zoo/models/yolo5l-voc-20classes_899-411aefb761eafaa3.pt",
    "yolov5x_voc_20": "http://download.deeplite.ai/zoo/models/yolo5x-voc-20classes_905-e8ddd018ae29751f.pt",
    "yolov5l_wider_face_8": "http://download.deeplite.ai/zoo/models/yolo5l-widerface-8cls-898_cdedd11381dbf565.pt",
    "yolov5m_wider_face_8": "http://download.deeplite.ai/zoo/models/yolo5m-widerface-8cls-878_8a99aaf8b8b9157b.pt",
    "yolov5l_voc_24": "http://download.deeplite.ai/zoo/models/yolo5l_voc-24_885_391dfc95d193faf5.pt",
    "yolov5m_voc_24": "http://download.deeplite.ai/zoo/models/yolo5m_voc_24_871_54be57d3f5a35a7b.pt",
    "yolov5_6s_voc_20": "http://download.deeplite.ai/zoo/models/yolo5_6s-voc-20classes_871-4ceb1b22b227c05c.pt",
    "yolov5_6n_voc_20": "http://download.deeplite.ai/zoo/models/yolo5_6n-voc-20classes_762-a6b8573a32ebb4c8.pt",
    "yolov5_6m_voc_20": "http://download.deeplite.ai/zoo/models/yolo5_6m-voc-20classes_902-50c151baffbf896e.pt",
    "yolov5_6s_coco_80": "http://download.deeplite.ai/zoo/models/yolov5_6s-coco-80classes_301-8ff1dabeec225366.pt",
    "yolov5_6m_coco_80": "http://download.deeplite.ai/zoo/models/yolov5_6m-coco-80classes_374-f93fa94b629c45ab.pt",
    "yolov5_6n_coco_80": "http://download.deeplite.ai/zoo/models/yolov5_6n-coco-80classes_211-e9e44a7de1f08ea2.pt",
    "yolov5_6s_person_detection_1": "http://download.deeplite.ai/zoo/models/yolov5_6s-person-detection-1class_738-9e9ac9dae14b0dcd.pt",
    "yolov5_6n_person_detection_1": "http://download.deeplite.ai/zoo/models/yolov5_6n-person-detection-1class_696-fff2a2c720e20752.pt",
}

yolov5_cfg = {
    "yolov5s": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5s.yaml",
    "yolov5m": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5m.yaml",
    "yolov5l": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5l.yaml",
    "yolov5x": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5x.yaml",
    "yolov5_6s": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5_6s.yaml",
    "yolov5_6m": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5_6m.yaml",
    "yolov5_6l": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5_6l.yaml",
    "yolov5_6x": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5_6x.yaml",
    "yolov5_6n": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov5_6n.yaml",
}

YOLOV5_MODELS = list(yolov5_cfg.keys())


def yolo5(
    net="yolov5s", _set_classes="voc_20", num_classes=20,
    pretrained=False, progress=True, device="cuda"
):
    config_path = get_project_root() / yolov5_cfg[net]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = model_urls[f"{net}_{_set_classes}"]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def yolo5_6(
    net="yolov5_6s", _set_classes="voc_20", num_classes=20,
    pretrained=False, progress=True, device="cuda"
):
    config_path = get_project_root() / yolov5_cfg[net]
    model = YoloV5_6(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = model_urls[f"{net}_{_set_classes}"]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


MODEL_TAG_TO_WRAPPER_FN_MAP = {
    "^yolov5[smlx]$": yolo5,
    "^yolov5_6[nsmlx]$": yolo5_6,
}

def make_wrapper_func(wrapper_name, net, _set_classes, num_classes):

    for net_name, model_fn in MODEL_TAG_TO_WRAPPER_FN_MAP.items():
        if re.match(net_name, net):
            model_wrapper_fn = model_fn

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
    'person_detection_1': ModelSet(1, ['yolov5_6n', 'yolov5_6s']),
    'voc_20': ModelSet(20, ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
        'yolov5_6n', 'yolov5_6s', 'yolov5_6m']),
    'voc_24': ModelSet(24, ['yolov5m', 'yolov5l']),
    'wider_face_8': ModelSet(8, ['yolov5m', 'yolov5l']),
    'coco_80': ModelSet(80, ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
        'yolov5_6n', 'yolov5_6s', 'yolov5_6m']),
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag.replace('v', ''), dataset]) # workaround for 'yolo5' -> 'yolov5' names
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
