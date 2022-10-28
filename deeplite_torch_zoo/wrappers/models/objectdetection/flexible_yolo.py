import urllib.parse as urlparse
from collections import namedtuple
from pathlib import Path

import deeplite_torch_zoo
from deeplite_torch_zoo.src.objectdetection.yolov5.flexible_yolo.model import \
    FlexibleYOLO
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

__all__ = []

def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]


CFG_PATH = 'deeplite_torch_zoo/src/objectdetection/yolov5/flexible_yolo/configs'
CHECKPOINT_STORAGE_URL = 'http://download.deeplite.ai/zoo/models/'

model_urls = {

}

model_configs = {
    'yolo_vgg': 'model_vgg.yaml',
    'yolo_shufflenet': 'model_shufflenet.yaml',
    'yolo_mobilenet': 'model_mobilenet.yaml',
    'yolo_resnet18': 'model_resnet.yaml',
    'yolo_resnet18x0.5': 'model_resnet.yaml',
    'yolo_resnet18x0.25': 'model_resnet.yaml',
    'yolo_resnet34': 'model_resnet.yaml',
    'yolo_resnet34x0.5': 'model_resnet.yaml',
    'yolo_resnet34x0.25': 'model_resnet.yaml',
    'yolo_resnet50': 'model_resnet.yaml',
    'yolo_resnet101': 'model_resnet.yaml',
    'yolo_resnet152': 'model_resnet.yaml',
}

model_kwargs = {
    'yolo_resnet18x0.25': {'backbone': {'version': 18, 'width': 0.25}, 'neck': None},
    'yolo_resnet18x0.5': {'backbone': {'version': 18, 'width': 0.5}, 'neck': None},
    'yolo_resnet18': {'backbone': {'version': 18}, 'neck': None},
    'yolo_resnet34': {'backbone': {'version': 34}, 'neck': None},
    'yolo_resnet34x0.25': {'backbone': {'version': 34, 'width': 0.25}, 'neck': None},
    'yolo_resnet34x0.5': {'backbone': {'version': 34, 'width': 0.5}, 'neck': None},
    'yolo_resnet50': {'backbone': {'version': 50}, 'neck': None},
    'yolo_resnet101': {'backbone': {'version': 101}, 'neck': None},
    'yolo_resnet152': {'backbone': {'version': 152}, 'neck': None},
}


def flexible_yolo(
    net='yolo_vgg', dataset_name='voc', num_classes=20,
    pretrained=False, progress=True, device='cuda'
):
    config_key = net
    config_path = get_project_root() / CFG_PATH / model_configs[config_key]
    backbone_kwargs, neck_kwargs = {}, {}
    if net in model_kwargs:
        backbone_kwargs, neck_kwargs = model_kwargs[net]['backbone'], model_kwargs[net]['neck']
    model = FlexibleYOLO(
        str(config_path),
        nc=num_classes,
        backbone_kwargs=backbone_kwargs,
        neck_kwargs=neck_kwargs,
    )
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f'{net}_{dataset_name}'])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def make_wrapper_func(wrapper_name, model_name, dataset_name, num_classes):
    model_wrapper_fn = flexible_yolo

    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection')
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device='cuda'):
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
    'voc': ModelSet(20, list(model_configs.keys())),
    'coco': ModelSet(80, list(model_configs.keys())),
}

for dataset, model_set in wrapper_funcs.items():
    for model_tag in model_set.model_list:
        name = '_'.join([model_tag, dataset])
        globals()[name] = make_wrapper_func(name, model_tag, dataset, model_set.num_classes)
        __all__.append(name)
