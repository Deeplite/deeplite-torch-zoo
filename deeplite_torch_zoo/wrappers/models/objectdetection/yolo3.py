
from collections import namedtuple

from deeplite_torch_zoo.src.objectdetection.yolov3.model.yolov3 import Yolov3
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights

__all__ = [
    "yolo3",
    "YOLOV3_MODELS",
]


model_urls = {
    "yolov3_voc_20": "http://download.deeplite.ai/zoo/models/yolo3-voc-0_839-a6149826183808aa.pth",
    "yolov3_voc_1": "http://download.deeplite.ai/zoo/models/yolov3-voc-1cls-0_888-1c73632fc187ef0c.pth",  # person
    "yolov3_voc_2": "http://download.deeplite.ai/zoo/models/yolov3-voc-2cls-0_911-b308f8a2686c19a6.pth",  # person and car
    "yolov3_lisa_11": "http://download.deeplite.ai/zoo/models/yolov3-lisa_11_830-663a0ec046402856.pth",
}

YOLOV3_MODELS = ["yolov3"]


def yolo3(
    net="yolov3", _set_classes="voc_20", num_classes=20, pretrained=False,
    progress=True, device="cuda", **kwargs
):
    model = Yolov3(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[f"yolov3_{_set_classes}"]
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
