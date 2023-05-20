import urllib.parse as urlparse
from pathlib import Path

import deeplite_torch_zoo
from deeplite_torch_zoo.src.objectdetection.flexible_yolo.model import YOLOv6
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

__all__ = []

def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]


CFG_PATH = 'deeplite_torch_zoo/src/objectdetection/flexible_yolo/yolov6/configs'
CHECKPOINT_STORAGE_URL = 'http://download.deeplite.ai/zoo/models/'

model_urls = {}

DATASET_LIST = [('person_detection', 1), ('voc', 20), ('coco', 80), ('voc07', 20), ('custom_person_detection', 1)]

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
    model_name='yolo6n', config_path=None, dataset_name='voc', pretrained=False, num_classes=20,
    progress=True, device='cuda', **kwargs
):
    model = YOLOv6(
        model_config=config_path,
        nc=num_classes,
        **kwargs
    )
    if pretrained:
        if f"{model_name}_{dataset_name}" not in model_urls:
            raise ValueError(f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. \n'
                              'Use pretrained=False if you want to create a untrained model.')
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f'{model_name}_{dataset_name}'])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def make_wrapper_func(wrapper_name, model_name, dataset_name, num_classes, config_path, **default_kwargs):
    model_wrapper_fn = yolov6
    has_checkpoint = True
    if f"{model_name}_{dataset_name}" not in model_urls:
        has_checkpoint = False

    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection', has_checkpoint=has_checkpoint)
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device='cuda', **kwargs):
        default_kwargs.update(**kwargs)
        return model_wrapper_fn(
            model_name=model_name,
            config_path=config_path,
            dataset_name=dataset_name,
            num_classes=num_classes,
            pretrained=pretrained,
            progress=progress,
            device=device,
            **default_kwargs
        )
    wrapper_func.__name__ = wrapper_name
    return wrapper_func


def get_model_scales():
    scale_dict = DEFAULT_MODEL_SCALES
    param_names = ('depth_mul', 'width_mul')
    return {cfg_name: dict(zip(param_names, param_cfg)) for cfg_name, param_cfg in scale_dict.items()}


full_model_dict = {}
for model_key, config_name in YOLOV6_CONFIGS.items():
    for cfg_name, param_dict in get_model_scales().items():
        full_model_dict[f'{model_key}-{cfg_name}'] = {
            'params': param_dict,
            'config': get_project_root() / CFG_PATH / config_name
        }


for dataset_tag, n_classes in DATASET_LIST:
    for model_tag, model_dict in full_model_dict.items():
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(name, model_tag, dataset_tag, n_classes,
                                                model_dict['config'], **model_dict['params'])
        __all__.append(name)
