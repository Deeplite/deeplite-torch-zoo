import urllib.parse as urlparse
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

model_urls = {}

model_configs = {
    'yolo_vgg16bn': 'model_vgg.yaml',
    'yolo_shufflenetv2': 'model_shufflenet.yaml',
    'yolo_mobilenetv3': 'model_mobilenet.yaml',
    'yolo_resnet18': 'model_resnet.yaml',
    'yolo_resnet18x0.5': 'model_resnet.yaml',
    'yolo_fdresnet18x0.5': 'model_resnet.yaml',
    'yolo_resnet18x0.25': 'model_resnet.yaml',
    'yolo_fdresnet18x0.25': 'model_resnet.yaml',
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
    'yolo_fdresnet18x0.25': {'backbone': {'version': 18, 'width': 0.25, 'first_block_downsampling': True}, 'neck': None},
    'yolo_fdresnet18x0.5': {'backbone': {'version': 18, 'width': 0.5, 'first_block_downsampling': True}, 'neck': None},
    'yolo_resnet18': {'backbone': {'version': 18}, 'neck': None},
    'yolo_resnet34': {'backbone': {'version': 34}, 'neck': None},
    'yolo_resnet34x0.25': {'backbone': {'version': 34, 'width': 0.25}, 'neck': None},
    'yolo_resnet34x0.5': {'backbone': {'version': 34, 'width': 0.5}, 'neck': None},
    'yolo_resnet50': {'backbone': {'version': 50}, 'neck': None},
    'yolo_resnet101': {'backbone': {'version': 101}, 'neck': None},
    'yolo_resnet152': {'backbone': {'version': 152}, 'neck': None},
    'yolo_vgg16bn': {'backbone': {'version': '16_bn'}, 'neck': None},
}


def flexible_yolo(
    model_name='yolo_resnet18', dataset_name='voc', num_classes=20,
    pretrained=False, progress=True, device='cuda'
):
    config_key = model_name
    config_path = get_project_root() / CFG_PATH / model_configs[config_key]
    backbone_kwargs, neck_kwargs = {}, {}
    if model_name in model_kwargs:
        backbone_kwargs, neck_kwargs = model_kwargs[model_name]['backbone'], model_kwargs[model_name]['neck']
    model = FlexibleYOLO(
        str(config_path),
        nc=num_classes,
        backbone_kwargs=backbone_kwargs,
        neck_kwargs=neck_kwargs,
    )
    if pretrained:
        if f"{model_name}_{dataset_name}" not in model_urls:
            raise ValueError(f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. \n'
                              'Use pretrained=False if you want to create a untrained model.')
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f'{model_name}_{dataset_name}'])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def make_wrapper_func(wrapper_name, model_name, dataset_name, num_classes):
    model_wrapper_fn = flexible_yolo
    has_checkpoint = True
    if f"{model_name}_{dataset_name}" not in model_urls:
        has_checkpoint = False

    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection', has_checkpoint=has_checkpoint)
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device='cuda'):
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


model_list = list(model_configs.keys())
datasets = [('person_detection', 1), ('voc', 20), ('coco', 80), ('voc07', 20), ('custom_person_detection', 1)]

for dataset_tag, n_classes in datasets:
    for model_tag in model_list:
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(name, model_tag, dataset_tag, n_classes)
        __all__.append(name)
