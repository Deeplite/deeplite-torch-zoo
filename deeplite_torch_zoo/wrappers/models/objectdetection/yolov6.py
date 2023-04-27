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

model_configs = {
    'yolo6n': 'yolov6n.py',
    'yolo6s': 'yolov6s.py',
    'yolo6m': 'yolov6m.py',
    'yolo6l': 'yolov6l.py',
}


def yolov6(
    model_name='yolo6n', dataset_name='voc', pretrained=False, num_classes=20,
    progress=True, device='cuda', **kwargs
):
    config_key = model_name
    config_path = get_project_root() / CFG_PATH / model_configs[config_key]

    model = YOLOv6(
        model_config=str(config_path),
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


def make_wrapper_func(wrapper_name, model_name, dataset_name, num_classes):
    model_wrapper_fn = yolov6
    has_checkpoint = True
    if f"{model_name}_{dataset_name}" not in model_urls:
        has_checkpoint = False

    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection', has_checkpoint=has_checkpoint)
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device='cuda', **kwargs):
        return model_wrapper_fn(
            model_name=model_name,
            dataset_name=dataset_name,
            num_classes=num_classes,
            pretrained=pretrained,
            progress=progress,
            device=device,
            **kwargs
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