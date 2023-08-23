from deeplite_torch_zoo.src.object_detection.yolo import YOLO
from deeplite_torch_zoo.api.models.object_detection.helpers import (
    make_wrapper_func, load_pretrained_model, get_project_root, DATASET_LIST
)

__all__ = []

CFG_PATH = 'deeplite_torch_zoo/src/object_detection/yolo/configs'

YOLO_CONFIGS = {
    'yolo3': 'yolo3/yolov3.yaml',
    'yolo4': 'yolo4/yolov4.yaml',
    'yolo5': 'yolo5/yolov5.yaml',
    'yolo7': 'yolo7/yolov7.yaml',
    'yolo8': 'yolo8/yolov8.yaml',
    ############################
    'yolo3-spp-': 'yolo3/yolov3-spp.yaml',
    'yolo3-tiny-': 'yolo3/yolov3-tiny.yaml',
    ############################
    'yolo4-tiny-': 'yolo4/yolov4-tiny.yaml',
    'yolo4-pacsp-': 'yolo4/yolov4-pacsp.yaml',
    'yolo4-csp-p5': 'yolo4/yolov4-csp-p5.yaml',
    'yolo4-csp-p6': 'yolo4/yolov4-csp-p6.yaml',
    'yolo4-csp-p7': 'yolo4/yolov4-csp-p7.yaml',
    ############################
    'yolo5.6': 'yolo5/yolov5.6.yaml',
    'yolo5-p2': 'yolo5/yolov5-p2.yaml',
    'yolo5-p34': 'yolo5/yolov5-p34.yaml',
    'yolo5-p6': 'yolo5/yolov5-p6.yaml',
    'yolo5-p7': 'yolo5/yolov5-p7.yaml',
    'yolo5-fpn-': 'yolo5/yolov5-fpn.yaml',
    'yolo5-bifpn-': 'yolo5/yolov5-bifpn.yaml',
    'yolo5-ghost-': 'yolo5/yolov5-ghost.yaml',
    'yolo5-panet-': 'yolo5/yolov5-panet.yaml',
    ############################
    'yolor': 'yolor/yolor-csp.yaml',
    'yolor-d6': 'yolor/yolor-d6.yaml',
    'yolor-e6': 'yolor/yolor-e6.yaml',
    'yolor-p6': 'yolor/yolor-p6.yaml',
    'yolor-w6': 'yolor/yolor-w6.yaml',
    'yolor-dwt-': 'yolor/yolor-dwt.yaml',
    'yolor-s2d-': 'yolor/yolor-s2d.yaml',
    ############################
    'yolo7-tiny-': 'yolo7/yolov7-tiny.yaml',
    'yolo7-e6': 'yolo7/yolov7-e6.yaml',
    'yolo7-e6e': 'yolo7/yolov7-e6e.yaml',
    'yolo7-w6': 'yolo7/yolov7-w6.yaml',
    ############################
    'yolo8-p2': 'yolo8/yolov8-p2.yaml',
    'yolo8-p6': 'yolo8/yolov8-p6.yaml',
    ############################
    'yolox': 'yolox/yolox.yaml',
    ############################
    'yolo-r50-csp-': 'misc/r50-csp.yaml',
    'yolo-x50-csp-': 'misc/x50-csp.yaml',  # to be fixed
    ############################
    'yolo-picodet-': 'picodet/yolo-picodet.yaml',
    'yolo5-lite-c-': 'yololite/yolov5_lite_c.yaml',
    'yolo5-lite-e-': 'yololite/yolov5_lite_e.yaml',
}

ACT_FN_TAGS = {'': None, '_relu': 'relu', '_hswish': 'hardswish'}

DEFAULT_MODEL_SCALES = {
    # [depth, width, max_channels]
    'n': [0.33, 0.25, 1024],
    's': [0.33, 0.50, 1024],
    'm': [0.67, 0.75, 1024],
    'l': [1.00, 1.00, 1024],
    'x': [1.00, 1.25, 1024],
    't': [0.25, 0.25, 1024],
    'd1w5': [1.0, 0.5, 1024],
    'd1w25': [1.0, 0.25, 1024],
    'd1w75': [1.0, 0.75, 1024],
    'd33w1': [0.33, 1.0, 1024],
    'd33w75': [0.33, 0.75, 1024],
    'd67w1': [0.67, 1.0, 1024],
    'd67w5': [0.67, 0.5, 1024],
    'd67w25': [0.67, 0.25, 1024],
}

V8_MODEL_SCALES = {
    'n': [0.33, 0.25, 1024],
    's': [0.33, 0.50, 1024],
    'm': [0.67, 0.75, 768],
    'l': [1.00, 1.00, 512],
    'x': [1.00, 1.25, 512],
    't': [0.25, 0.25, 1024],
    'd1w5': [1.0, 0.5, 512],
    'd1w25': [1.0, 0.25, 512],
    'd1w75': [1.0, 0.75, 512],
    'd33w1': [0.33, 1.0, 1024],
    'd33w75': [0.33, 0.75, 1024],
    'd67w1': [0.67, 1.0, 768],
    'd67w5': [0.67, 0.5, 768],
    'd67w25': [0.67, 0.25, 768],
}

CUSTOM_MODEL_SCALES = {'yolo8': V8_MODEL_SCALES}


def create_yolo_model(
    model_name="yolo5s",
    dataset_name="voc",
    num_classes=20,
    config_path=None,
    pretrained=False,
    **kwargs,
):  # pylint: disable=W0621
    model = YOLO(
        config_path,
        nc=num_classes,
        **kwargs,
    )
    if pretrained:
        model = load_pretrained_model(model, model_name, dataset_name)
    return model


def get_model_scales(_model_key):
    scale_dict = CUSTOM_MODEL_SCALES.get(_model_key, DEFAULT_MODEL_SCALES)
    param_names = ('depth_mul', 'width_mul', 'max_channels')
    return {
        cfg_name: dict(zip(param_names, param_cfg))
        for cfg_name, param_cfg in scale_dict.items()
    }


full_model_dict = {}
for model_key, config_name in YOLO_CONFIGS.items():
    for cfg_name, param_dict in get_model_scales(model_key).items():
        for activation_fn_tag, act_fn_name in ACT_FN_TAGS.items():
            full_model_dict[f'{model_key}{cfg_name}{activation_fn_tag}'] = {
                'params': {**param_dict, 'activation_type': act_fn_name},
                'config': get_project_root() / CFG_PATH / config_name,
            }


for dataset_tag, n_classes in DATASET_LIST:
    for model_tag, model_dict in full_model_dict.items():
        name = '_'.join([model_tag, dataset_tag])
        globals()[name] = make_wrapper_func(
            create_yolo_model,
            name,
            model_tag,
            dataset_tag,
            n_classes,
            config_path=model_dict['config'],
            **model_dict['params'],
        )
        __all__.append(name)
