import re
import urllib.parse as urlparse
from functools import partial
from pathlib import Path

import deeplite_torch_zoo
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_6 import \
    YOLOModel
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

__all__ = []

def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]

CFG_PATH = "deeplite_torch_zoo/src/objectdetection/yolov5/configs/model_configs/yolo5"
CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/"

model_urls = {
    "yolo5_6s_coco": "yolov5_6s-coco-80classes_301-8ff1dabeec225366.pt",
    "yolo5_6m_coco": "yolov5_6m-coco-80classes_374-f93fa94b629c45ab.pt",
    "yolo5_6n_coco": "yolov5_6n-coco-80classes_211-e9e44a7de1f08ea2.pt",
    "yolo5_6sa_coco": "yolov5_6sa-coco-80classes_297-6c1972b5f7ae6ab6.pt",
    "yolo5_6ma_coco": "yolov5_6ma-coco-80classes_365-4756729c4f6a834f.pt",
    "yolo5_6n_hswish_coco": "yolov5_6n_hswish-coco-80classes-183-a2fed163ec98352a.pt",
    "yolo5_6n_relu_coco": "yolov5_6n_relu-coco-80classes-167-7b6609497c63df79.pt",
}

voc_model_urls = {
    "yolo5_6n_voc": "yolo5_6n-voc-20classes_762-a6b8573a32ebb4c8.pt",
    "yolo5_6s_voc": "yolo5_6s-voc-20classes_871-4ceb1b22b227c05c.pt",
    "yolo5_6m_voc": "yolo5_6m-voc-20classes_902-50c151baffbf896e.pt",
    "yolo5_6l_voc": "yolov5_6l-voc-20classes_875_3fb90f0c405f170c.pt",
    "yolo5_6x_voc": "yolov5_6x-voc-20classes_884_a2b6fb7234218cf6.pt",
    "yolo5_6n_voc07": "yolov5_6n-voc07-20classes-620_037230667eff7b12.pt",
    "yolo5_6s_voc07": "yolov5_6s-voc07-20classes-687_4d221fd4edc09ce1.pt",
    "yolo5_6s_relu_voc": "yolov5_6s_relu-voc-20classes-819_a35dff53b174e383.pt",
    "yolo5_6m_relu_voc": "yolov5_6m_relu-voc-20classes-856_c5c23135e6d5012f.pt",
}

person_detection_model_urls = {
    "yolo5_6n_person_detection": "yolov5_6n-person-detection-1class_696-fff2a2c720e20752.pt",
    "yolo5_6s_person_detection": "yolov5_6s-person-detection-1class_738-9e9ac9dae14b0dcd.pt",
    "yolo5_6n_relu_person_detection": "yolov5_6n_relu-person-detection-1class_621-6794298f12d33ba8.pt",
    "yolo5_6s_relu_person_detection": "yolov5_6s_relu-person-detection-1class_682-45ae979a06b80767.pt",
    "yolo5_6m_relu_person_detection": "yolov5_6m_relu-person-detection-1class_709-3f59321c540d2d1c.pt",
    "yolo5_6sa_person_detection": "yolov5_6sa-person-detection-1class_659_015807ae6899af0f.pt",
    "yolo5_6n_relu_custom_person_detection": "yolov5_6n_relu-person-detection-640x-1class_889-d1e995d6639a52de.pt",
    "yolo5_6n_tiny_relu_custom_person_detection": "yolov5_6n_tiny_relu-person-detection-640x-1class_875-4ff20f9a76eadbcb.pt",
    "yolo5_6n_tiny_hswish_custom_person_detection": "yolov5_6n_tiny_hs-person-detection-640x-1class_891-2e833a6c98108add.pt",
}

model_urls.update(voc_model_urls)
model_urls.update(person_detection_model_urls)

yolov5_cfg = {
    "yolo5_6s": "yolov5_6s.yaml",
    "yolo5_6m": "yolov5_6m.yaml",
    "yolo5_6l": "yolov5_6l.yaml",
    "yolo5_6x": "yolov5_6x.yaml",
    "yolo5_6n": "yolov5_6n.yaml",
    "yolo5_6sa": "yolov5_6sa.yaml",
    "yolo5_6ma": "yolov5_6ma.yaml",
    "yolo5_6n_tiny": "yolov5_6n_tiny.yaml"
}

MODEL_NAME_SUFFICES = ('relu', 'hswish')

def yolo5_6(
    model_name="yolo5_6s", dataset_name="voc", num_classes=20, activation_type=None,
    pretrained=False, progress=True, device="cuda", ch=3,
):  # pylint: disable=W0621
    if 'yolo5_6' not in model_name and 'yolo5' in model_name:
        model_name = model_name.replace('yolo5', 'yolo5_6')

    config_key = model_name
    for suffix in MODEL_NAME_SUFFICES:
        config_key = re.sub(f'\_{suffix}$', '', config_key) # pylint: disable=W1401
    config_path = get_project_root() / CFG_PATH / yolov5_cfg[config_key]
    model = YOLOModel(config_path, ch=ch, nc=num_classes, activation_type=activation_type)
    if pretrained:
        if f"{model_name}_{dataset_name}" not in model_urls:
            raise ValueError(f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. \n'
                              'Use pretrained=False if you want to create a untrained model.')
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{model_name}_{dataset_name}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


MODEL_TAG_TO_WRAPPER_FN_MAP = {
    "^yolo5_6[nsmlx]$": yolo5_6,
    "^yolo5_6[nsmlx]_relu$": partial(yolo5_6, activation_type="relu"),
    "^yolo5_6[nsmlx]_hswish$": partial(yolo5_6, activation_type="hardswish"),
    "^yolo5_6[nsmlx]a$": yolo5_6,
    "^yolo5_6[nsmlx]a_relu$": partial(yolo5_6, activation_type="relu"),
    "^yolo5_6[nsmlx]a_hswish$": partial(yolo5_6, activation_type="hardswish"),
    "^yolo5_6[nsmlx]_tiny$": yolo5_6,
    "^yolo5_6[nsmlx]_tiny_relu$": partial(yolo5_6, activation_type="relu"),
    "^yolo5_6[nsmlx]_tiny_hswish$": partial(yolo5_6, activation_type="hardswish"),
}

def make_wrapper_func(wrapper_name, model_name, dataset_name, num_classes):  # pylint: disable=W0621

    model_wrapper_fn = None
    for net_name, model_fn in MODEL_TAG_TO_WRAPPER_FN_MAP.items():
        if re.match(net_name, model_name):
            model_wrapper_fn = model_fn
        if re.match(net_name, model_name.replace('yolo5', 'yolo5_6')):
            model_wrapper_fn = model_fn
    if model_wrapper_fn is None:
        raise ValueError(f'Could not find a wrapper function for model name {model_name}')
    has_checkpoint = True
    if f"{model_name}_{dataset_name}" not in model_urls:
        has_checkpoint = False

    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='object_detection', has_checkpoint=has_checkpoint)
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device="cuda"):
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


model_list = list(yolov5_cfg.keys())
for model_name_suffix in MODEL_NAME_SUFFICES:
    model_list += [f'{model_name}_{model_name_suffix}' for model_name in yolov5_cfg]

for model_name in yolov5_cfg:
    if 'yolo5_6' in model_name:
        model_list.append(model_name.replace('yolo5_6', 'yolo5'))

datasets = [('person_detection', 1), ('voc', 20), ('coco', 80), ('voc07', 20), ('custom_person_detection', 1)]

for dataset_tag, n_classes in datasets:
    for model_tag in model_list:
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(name, model_tag, dataset_tag, n_classes)
        __all__.append(name)
