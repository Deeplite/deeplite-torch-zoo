# Ultralytics YOLO 🚀, AGPL-3.0 license

import pathlib
from collections import namedtuple
from functools import partial

from addict import Dict

from deeplite_torch_zoo.utils import RANK
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY

from deeplite_torch_zoo.src.object_detection.datasets.dataloader import get_dataloader
from deeplite_torch_zoo.api.datasets.object_detection.utils import check_det_dataset


__all__ = []

HERE = pathlib.Path(__file__).parent

CONFIG_VARS = ['rect', 'cache', 'single_cls', 'task', 'classes', 'fraction',
               'mosaic', 'mixup', 'mask_ratio', 'overlap_mask', 'copy_paste', 'degrees',
               'translate', 'scale', 'shear', 'perspective', 'fliplr', 'flipud', 'hsv_h', 'hsv_s', 'hsv_v']

DEFAULT_IMG_RES = 640
DatasetConfig = namedtuple('DatasetConfig', ['yaml_file', 'num_classes', 'default_res'])
DATASET_CONFIGS = {
    'voc': DatasetConfig('VOC.yaml', 20, 448),
    'coco': DatasetConfig('coco.yaml', 80, 640),
    'coco8': DatasetConfig('coco8.yaml', 80, 640),
    'coco128': DatasetConfig('coco128.yaml', 80, 640),
    'SKU-110K': DatasetConfig('SKU-110K.yaml', 1, 640),
}


def create_detection_dataloaders(
    data_root=None,
    dataset_config=None,
    batch_size=64,
    image_size=None,
    num_workers=8,
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
    gs=32,
):
    cfg = Dict()
    for var_name in CONFIG_VARS:
        cfg[var_name] = locals()[var_name]

    if image_size is None:
        if dataset_config in DATASET_CONFIGS:
            image_size = DATASET_CONFIGS[dataset_config].default_res
        else:
            image_size = DEFAULT_IMG_RES

    cfg.imgsz = image_size
    cfg.workers = num_workers

    if dataset_config.endswith('.yaml'):
        data = check_det_dataset(dataset_config, data_root=data_root)
    elif dataset_config in DATASET_CONFIGS:
        data = check_det_dataset(HERE / 'configs' / DATASET_CONFIGS[dataset_config].yaml_file, data_root=data_root)
    else:
        raise ValueError(
            f'Incorrect dataset name/config passed: {dataset_config}. Either pass a path '
            f'to a YAML config or a valid zoo dataset name. Supported datasets: {DATASET_CONFIGS.keys()}'
        )

    trainset, testset = data['train'], data.get('val') or data.get('test')
    train_loader = get_dataloader(trainset, data, cfg, batch_size=batch_size, rank=RANK, mode='train', gs=gs)
    test_loader = get_dataloader(testset, data, cfg, batch_size=batch_size * 2, rank=-1, mode='val', gs=gs)
    return {'train': train_loader, 'test': test_loader}


for dataset_name_key in DATASET_CONFIGS:
    wrapper_fn_name = f'get_{dataset_name_key}_for_yolo'
    wrapper_fn = partial(create_detection_dataloaders, dataset_config=dataset_name_key)
    globals()[wrapper_fn_name] = wrapper_fn
    DATASET_WRAPPER_REGISTRY.register(dataset_name=dataset_name_key)(wrapper_fn)
    __all__.append(wrapper_fn_name)
