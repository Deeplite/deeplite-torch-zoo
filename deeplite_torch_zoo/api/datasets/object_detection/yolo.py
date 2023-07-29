# Ultralytics YOLO 🚀, AGPL-3.0 license

import pathlib
from functools import partial

from addict import Dict

from deeplite_torch_zoo.utils import RANK
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY

from deeplite_torch_zoo.src.object_detection.datasets.dataloader import get_dataloader
from deeplite_torch_zoo.api.datasets.object_detection.utils import check_det_dataset


__all__ = []

HERE = pathlib.Path(__file__).parent

CONFIG_VARS = ['workers', 'rect', 'cache', 'single_cls', 'task', 'classes', 'fraction',
               'mosaic', 'mixup', 'mask_ratio', 'overlap_mask', 'copy_paste', 'degrees',
               'translate', 'scale', 'shear', 'perspective', 'fliplr', 'flipud', 'hsv_h', 'hsv_s', 'hsv_v']

DATASET_CONFIGS = {
    'voc': 'VOC.yaml',
    'coco': 'coco.yaml',
    'coco8': 'coco8.yaml',
    'coco128': 'coco128.yaml',
    'SKU-110K': 'SKU-110K.yaml',
}

DEFAULT_RESOLUTIONS = {
    'voc': 448,
    'coco': 640,
    'coco8': 640,
    'coco128': 640,
    'SKU-110K': 640,
}


def create_detection_dataloaders(
    data_root=None,
    dataset_config=None,
    batch_size=64,
    image_size=None,
    workers=8,
    rect=False,
    cache=False,
    single_cls=False,
    task='detect',
    classes=None,
    fraction=1.0,
    mosaic=1.0,
    mixup=0.0,
    mask_ratio=4,
    overlap_mask=True,
    copy_paste=0.0,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
):
    cfg = Dict()
    for var_name in CONFIG_VARS:
        cfg[var_name] = locals()[var_name]

    if image_size is None:
        image_size = DEFAULT_RESOLUTIONS.get(dataset_config, 640)
    cfg.imgsz = image_size

    if dataset_config.endswith('.yaml'):
        data = check_det_dataset(dataset_config)
    elif dataset_config in DATASET_CONFIGS:
        data = check_det_dataset(HERE / 'configs' / DATASET_CONFIGS[dataset_config])
    else:
        raise ValueError

    trainset, testset = data['train'], data.get('val') or data.get('test')
    train_loader = get_dataloader(trainset, data, cfg, batch_size=batch_size, rank=RANK, mode='train')
    test_loader = get_dataloader(testset, data, cfg, batch_size=batch_size * 2, rank=-1, mode='val')
    return {'train': train_loader, 'test': test_loader}


for dataset_name_key in DATASET_CONFIGS:
    wrapper_fn_name = f'get_{dataset_name_key}_for_yolo'
    wrapper_fn = partial(create_detection_dataloaders, dataset_config=dataset_name_key)
    globals()[wrapper_fn_name] = wrapper_fn
    DATASET_WRAPPER_REGISTRY.register(dataset_name=dataset_name_key)(wrapper_fn)
    __all__.append(wrapper_fn_name)
