from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.model import FlexibleYOLO
from deeplite_torch_zoo.api.models.object_detection.helpers import (
    make_wrapper_func, get_project_root, load_pretrained_model
)
from deeplite_torch_zoo.api.datasets.object_detection.yolo import DATASET_CONFIGS


__all__ = []

CFG_PATH = 'deeplite_torch_zoo/src/object_detection/yolo/flexible_yolo/configs'

model_configs = {
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
    'yolo_resnet18_rtdetr':'model_resnet_rtdetr.yaml',
}

model_kwargs = {
    'yolo_resnet18x0.25': {'backbone': {'version': 18, 'width': 0.25}, 'neck': None},
    'yolo_resnet18x0.5': {'backbone': {'version': 18, 'width': 0.5}, 'neck': None},
    'yolo_fdresnet18x0.25': {
        'backbone': {'version': 18, 'width': 0.25, 'first_block_downsampling': True},
        'neck': None,
    },
    'yolo_fdresnet18x0.5': {
        'backbone': {'version': 18, 'width': 0.5, 'first_block_downsampling': True},
        'neck': None,
    },
    'yolo_resnet18': {'backbone': {'version': 18}, 'neck': None},
    'yolo_resnet34': {'backbone': {'version': 34}, 'neck': None},
    'yolo_resnet34x0.25': {'backbone': {'version': 34, 'width': 0.25}, 'neck': None},
    'yolo_resnet34x0.5': {'backbone': {'version': 34, 'width': 0.5}, 'neck': None},
    'yolo_resnet50': {'backbone': {'version': 50}, 'neck': None},
    'yolo_resnet101': {'backbone': {'version': 101}, 'neck': None},
    'yolo_resnet152': {'backbone': {'version': 152}, 'neck': None},
    'yolo_resnet18_rtdetr': {'backbone': {'version': 18, 'width': 0.25}, 'neck': None},
}

def flexible_yolo(
    model_name='yolo_resnet18',
    dataset_name='voc',
    num_classes=20,
    pretrained=False,
    **kwargs,
):
    config_key = model_name
    config_path = get_project_root() / CFG_PATH / model_configs[config_key]
    backbone_kwargs, neck_kwargs = {}, {}
    if model_name in model_kwargs:
        backbone_kwargs, neck_kwargs = (
            model_kwargs[model_name]['backbone'],
            model_kwargs[model_name]['neck'],
        )
    model = FlexibleYOLO(
        str(config_path),
        nc=num_classes,
        backbone_kwargs=backbone_kwargs,
        neck_kwargs=neck_kwargs,
        **kwargs,
    )
    if pretrained:
        model = load_pretrained_model(model, model_name, dataset_name)
    return model


model_list = list(model_configs.keys())
for dataset_tag, dataset_config in DATASET_CONFIGS.items():
    for model_tag in model_list:
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(flexible_yolo, name, model_tag, dataset_tag,
                                            dataset_config.num_classes)
        __all__.append(name)
