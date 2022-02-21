import re
from pathlib import Path
from collections import namedtuple

from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5 import YoloV5
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import YoloV5_6
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY



def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent


__all__ = [
    "yolo4",
    "yolo4_6",
    "YOLOV4_MODELS",
]

model_urls = {
    "yolov4s_voc_20": "http://download.deeplite.ai/zoo/models/yolo4s-voc-20classes_850-270ddc5d43290a95.pth",
    "yolov4m_voc_20": "http://download.deeplite.ai/zoo/models/yolo4m-voc-20classes_885-b854caad9ca7fb7c.pth",
    "yolov4l_voc_20": "http://download.deeplite.ai/zoo/models/yolo4l-voc-20classes_872-9f54132ce2934fbf.pth",
    "yolov4x_voc_20": "http://download.deeplite.ai/zoo/models/yolo4x-voc-20classes_882-187f352b9d0d29c6.pth",
    "yolov4m_lisa_11": "http://download.deeplite.ai/zoo/models/yolov4m-lisa_11_880-6615c5e27557fab0.pth",
    "yolov4l_leaky_voc_20": "http://download.deeplite.ai/zoo/models/yolo4l-leaky-voc-20classes_891-2c0f78ee3938ade3.pt",
    "yolov4_6s_coco_80": "http://download.deeplite.ai/zoo/models/yolov4_6s-coco-80classes-288_b112910223d6c56d.pt",
    "yolov4_6m_coco_80": "http://download.deeplite.ai/zoo/models/yolov4_6m-coco-80classes-309_02b2013002a4724b.pt",
}

yolov4_cfg = {
    "yolov4s": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4s-mish.yaml",
    "yolov4m": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4m-mish.yaml",
    "yolov4l": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4l-mish.yaml",
    "yolov4x": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4x-mish.yaml",
    "yolov4l_leaky": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4l-leaky.yaml",
    "yolov4_6s": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4s-mish.yaml",
    "yolov4_6m": "deeplite_torch_zoo/src/objectdetection/configs/model_configs/yolov4m-mish.yaml",
}

YOLOV4_MODELS = list(yolov4_cfg.keys())


def yolo4(
    net="yolov4s", dataset_name="voc_20", num_classes=20, pretrained=False,
    progress=True, device="cuda",
):
    config_path = get_project_root() / yolov4_cfg[net]
    model = YoloV5(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = model_urls[f"{net}_{dataset_name}"]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def yolo4_6(
    net="yolov4s", dataset_name="voc_20", num_classes=20, pretrained=False,
    progress=True, device="cuda",
):
    config_path = get_project_root() / yolov4_cfg[net]
    model = YoloV5_6(config_path, ch=3, nc=num_classes)
    if pretrained:
        checkpoint_url = model_urls[f"{net}_{dataset_name}"]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


MODEL_TAG_TO_WRAPPER_FN_MAP = {
    "^yolov4[smlx]$": yolo4,
    "^yolov4_6[nsmlx]$": yolo4_6,
    "^yolov4[smlx]_leaky$": yolo4,
}


def make_wrapper_func(wrapper_name, net, dataset_name, num_classes):
    for net_name, model_fn in MODEL_TAG_TO_WRAPPER_FN_MAP.items():
        if re.match(net_name, net):
            model_wrapper_fn = model_fn

    model_name = net.replace('v', '')

    @MODEL_WRAPPER_REGISTRY.register(model_name, dataset_name)
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
    'voc_20': ModelSet(20, ['yolov4s', 'yolov4m', 'yolov4l', 'yolov4x', 'yolov4l_leaky']),
    'coco_80': ModelSet(80, ['yolov4_6s', 'yolov4_6m']),
    'lisa_11': ModelSet(11, ['yolov4m'])
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag.replace('v', ''), dataset]) # workaround for 'yolo4' -> 'yolov4' names
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
