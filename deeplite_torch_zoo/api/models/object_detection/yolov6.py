from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6_model import YOLOv6
from deeplite_torch_zoo.api.models.object_detection.helpers import (
    make_wrapper_func, get_project_root, load_pretrained_model, DATASET_LIST
)

__all__ = []

CFG_PATH = 'deeplite_torch_zoo/src/object_detection/yolo/flexible_yolo/yolov6/configs'

YOLOV6_CONFIGS = {
    'yolo6s': 'yolov6s.py',
    'yolo6m': 'yolov6m.py',
    'yolo6l': 'yolov6l.py',
}

DEFAULT_MODEL_SCALES = {
    # [depth, width]
    'd33w25': [0.33, 0.25],
    'd33w5': [0.33, 0.50],
    'd6w75': [0.6, 0.75],
    'd1w1': [1.00, 1.00],
    'd1w5': [1.0, 0.5],
    'd1w25': [1.0, 0.25],
    'd1w75': [1.0, 0.75],
    'd33w1': [0.33, 1.0],
    'd33w75': [0.33, 0.75],
    'd6w1': [0.6, 1.0],
    'd6w5': [0.6, 0.5],
    'd6w25': [0.6, 0.25],
}


def yolov6(
    model_name='yolo6n',
    config_path=None,
    dataset_name='voc',
    pretrained=False,
    num_classes=20,
    **kwargs,
):
    model = YOLOv6(model_config=config_path, nc=num_classes, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, model_name, dataset_name)
    return model


def get_model_scales():
    scale_dict = DEFAULT_MODEL_SCALES
    param_names = ('depth_mul', 'width_mul')
    return {
        cfg_name: dict(zip(param_names, param_cfg))
        for cfg_name, param_cfg in scale_dict.items()
    }


full_model_dict = {}
for model_key, config_name in YOLOV6_CONFIGS.items():
    for cfg_name, param_dict in get_model_scales().items():
        full_model_dict[f'{model_key}-{cfg_name}'] = {
            'params': param_dict,
            'config': get_project_root() / CFG_PATH / config_name,
        }


for dataset_tag, n_classes in DATASET_LIST:
    for model_tag, model_dict in full_model_dict.items():
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(
            yolov6,
            name,
            model_tag,
            dataset_tag,
            n_classes,
            config_path=model_dict['config'],
            **model_dict['params'],
        )
        __all__.append(name)
