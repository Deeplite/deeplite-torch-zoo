from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolo_nas_model import YOLONAS
from deeplite_torch_zoo.api.models.object_detection.helpers import (
    make_wrapper_func, load_pretrained_model
)
from deeplite_torch_zoo.api.datasets.object_detection.yolo import DATASET_CONFIGS


__all__ = []


YOLONAS_CONFIGS = {
    'yolonas_l': 'yolo_nas_l_arch_params',
    'yolonas_m': 'yolo_nas_m_arch_params',
    'yolonas_s': 'yolo_nas_s_arch_params',
}


def yolonas(
    model_name='yolonas_s',
    dataset_name='voc',
    pretrained=False,
    num_classes=20,
    **kwargs,
):
    model = YOLONAS(arch_name=YOLONAS_CONFIGS[model_name], nc=num_classes, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, model_name, dataset_name)
    return model


for dataset_tag, dataset_config in DATASET_CONFIGS.items():
    for model_tag in YOLONAS_CONFIGS:
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(
            yolonas,
            name,
            model_tag,
            dataset_tag,
            dataset_config.num_classes,
        )
        __all__.append(name)
