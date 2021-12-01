
from collections import namedtuple

from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.objectdetection.yolov3.model.yolov3 import Yolov3


__all__ = [
    "yolo3",
    "YOLOV3_MODELS",
]


model_urls = {
    "yolov3_voc_20": "http://download.deeplite.ai/zoo/models/yolo3-voc-0_839-a6149826183808aa.pth",
    "yolov3_voc_01": "http://download.deeplite.ai/zoo/models/yolov3-voc-1cls-0_888-1c73632fc187ef0c.pth",  # person
    "yolov3_voc_02": "http://download.deeplite.ai/zoo/models/yolov3-voc-2cls-0_911-b308f8a2686c19a6.pth",  # person and car
    "yolov3_lisa_11": "http://download.deeplite.ai/zoo/models/yolov3-lisa_11_830-663a0ec046402856.pth",
}

YOLOV3_MODELS = ["yolov3"]


def yolo3(
    _set_classes="voc_20", num_classes=20, pretrained=False,
    progress=True, device="cuda", skip_loading_last_layer=False, **kwargs
):
    model = Yolov3(num_classes=num_classes)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(
            model_urls[f"yolov3_{_set_classes}"],
            progress=progress,
            check_hash=True,
            map_location=device,
        )
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model.to(device)


def make_wrapper_func(name, net, _set_classes, num_classes):
    def wrapper_func(pretrained=False, progress=True, device="cuda"):
        return yolo3(
            net=net,
            _set_classes=_set_classes,
            num_classes=num_classes,
            pretrained=pretrained,
            progress=progress,
            device=device,
        )
    wrapper_func.__name__ = name
    return wrapper_func


ModelSet = namedtuple('ModelSet', ['num_classes', 'model_list'])
wrapper_funcs = {
    'voc_20': ModelSet(20, ['yolov3']),
    'voc_01': ModelSet(1, ['yolov3']),
    'voc_02': ModelSet(2, ['yolov3']),
    'lisa_11': ModelSet(11, ['yolov3']),
}

for dataset in wrapper_funcs:
    for net in wrapper_funcs[dataset].model_list:
        name = '_'.join([net.replace('v', ''), dataset]) # workaround for 'yolo3' -> 'yolov3' names
        globals()[name] = make_wrapper_func(name, net, dataset, wrapper_funcs[dataset].num_classes)
        __all__.append(name)
