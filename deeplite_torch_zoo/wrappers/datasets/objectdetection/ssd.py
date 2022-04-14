import os
import sys
from functools import partial

from torch.utils.data import ConcatDataset

from deeplite_torch_zoo.wrappers.datasets.utils import get_dataloader
from deeplite_torch_zoo.src.objectdetection.ssd.datasets.voc_dataset import VOCDataset
from deeplite_torch_zoo.src.objectdetection.ssd.datasets.coco import CocoDetectionBoundingBox
from deeplite_torch_zoo.src.objectdetection.ssd.datasets.wider_face import WiderFace
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.data_preprocessing import TrainAugmentation
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.ssd import MatchPrior
from deeplite_torch_zoo.src.objectdetection.ssd.config.vgg_ssd_config import VGG_CONFIG
from deeplite_torch_zoo.src.objectdetection.ssd.config.mobilenetv1_ssd_config import (
    MOBILENET_CONFIG,
)
from deeplite_torch_zoo.src.objectdetection.datasets.coco_config import COCO_MISSING_IDS, COCO_DATA_CATEGORIES
from deeplite_torch_zoo.wrappers.registries import DATA_WRAPPER_REGISTRY


__all__ = []


def make_dataset_wrapper(wrapper_name, dataset_create_fn):
    def wrapper_func(data_root, config, batch_size=32, num_workers=4,
        fp16=False, distributed=False, device="cuda", **kwargs):

        if len(kwargs):
            print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

        train_dataset, test_dataset = dataset_create_fn(data_root, config, **kwargs)

        train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
            fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)
        test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
            fp16=fp16, distributed=distributed, shuffle=False, device=device)

        return {"train": train_loader, "val": test_loader, "test": test_loader}

    wrapper_func.__name__ = wrapper_name
    return wrapper_func


def create_voc_datasets(data_root, config, num_classes):
    train_dataset_07, test_dataset = _get_voc_for_ssd(
        data_root=os.path.join(data_root, "VOC2007"), config=config, num_classes=num_classes
    )
    train_dataset_12, _ = _get_voc_for_ssd(
        data_root=os.path.join(data_root, "VOC2012"), config=config, num_classes=num_classes
    )
    train_dataset = ConcatDataset([train_dataset_07, train_dataset_12])
    return train_dataset, test_dataset


def _get_voc_for_ssd(data_root, config=None, num_classes=21):
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

    train_dataset = VOCDataset(root=data_root, n_classes=num_classes, is_test=False,
        transform=train_transform, target_transform=target_transform)

    if "VOC2012" not in data_root:
        test_dataset = VOCDataset(root=data_root, n_classes=num_classes, is_test=True)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def create_coco_datasets(data_root, config=None, img_size=416, train_ann_file=None, train_dir=None,
    val_ann_file=None, val_dir=None, missing_ids=[], classes=[]):

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

    train_annotate = os.path.join(data_root, train_ann_file)
    train_coco_root = os.path.join(data_root, train_dir)
    train_dataset = CocoDetectionBoundingBox(train_coco_root, train_annotate, transform=train_transform,
        target_transform=target_transform, missing_ids=missing_ids, classes=classes)

    val_annotate = os.path.join(data_root, val_ann_file)
    val_coco_root = os.path.join(data_root, val_dir)
    test_dataset = CocoDetectionBoundingBox(val_coco_root, val_annotate, missing_ids=missing_ids, classes=classes)

    return train_dataset, test_dataset


def create_widerface_datasets(data_root, config=None):
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

    train_dataset = WiderFace(root=data_root, split="train",
        transform=train_transform, target_transform=target_transform)
    test_dataset = WiderFace(root=data_root, split="val")
    return train_dataset, test_dataset


DATASET_WRAPPER_FNS = {
    'coco': create_coco_datasets,
    'voc': create_voc_datasets,
    'wider_face': create_widerface_datasets,
}

for dataset_name_key, wrapper_fn in DATASET_WRAPPER_FNS.items():
    wrapper_fn_name = f'get_{dataset_name_key}_for_ssd'
    func = make_dataset_wrapper(wrapper_fn_name, dataset_create_fn=wrapper_fn)
    globals()[wrapper_fn_name] = func
    DATA_WRAPPER_REGISTRY.register(dataset_name=dataset_name_key, model_type='ssd')(func)
    __all__.append(wrapper_fn_name)


VOC_DATASET_MODEL_WRAPPERS = {
    'mb1_ssd': MOBILENET_CONFIG(),
    'mb2_ssd_lite': MOBILENET_CONFIG(),
    'resnet18_ssd': VGG_CONFIG(),
    'resnet34_ssd': VGG_CONFIG(),
    'resnet50_ssd': VGG_CONFIG(),
    'vgg16_ssd': VGG_CONFIG(),
}

for model_name_key, model_config in VOC_DATASET_MODEL_WRAPPERS.items():
    wrapper_fn_name = f'get_voc_for_{model_name_key}'
    wrapper_fn = partial(globals()['get_voc_for_ssd'],
        config=model_config, num_classes=21)
    globals()[wrapper_fn_name] = wrapper_fn
    DATA_WRAPPER_REGISTRY.register(dataset_name='voc', model_type=model_name_key)(wrapper_fn)
    __all__.append(wrapper_fn_name)


WIDERFACE_DATASET_MODEL_WRAPPERS = {
    'vgg16_ssd': VGG_CONFIG(),
}

for model_name_key, model_config in WIDERFACE_DATASET_MODEL_WRAPPERS.items():
    wrapper_fn_name = f'get_wider_face_for_{model_name_key}'
    wrapper_fn = partial(globals()['get_wider_face_for_ssd'],
        config=model_config, num_classes=21)
    globals()[wrapper_fn_name] = wrapper_fn
    DATA_WRAPPER_REGISTRY.register(dataset_name='wider_face', model_type=model_name_key)(wrapper_fn)
    __all__.append(wrapper_fn_name)


COCO_DATASET_MODEL_WRAPPERS = {
    ('coco', 'mb2_ssd'): {
        'config': MOBILENET_CONFIG(),
        'train_ann_file': "annotations/instances_train2017.json",
        'train_dir': "train2017",
        'val_ann_file': "annotations/instances_val2017.json",
        'val_dir': "val2017",
        'classes': COCO_DATA_CATEGORIES["CLASSES"],
        'missing_ids': COCO_MISSING_IDS,
    },
    ('coco_gm', 'mb2_ssd'): {
        'config': MOBILENET_CONFIG(),
        'train_ann_file': "train_data_COCO.json",
        'train_dir': "images/train",
        'val_ann_file': "test_data_COCO.json",
        'val_dir': "images/test",
        'classes': ["class1", "class2", "class3", "class4", "class5", "class6"],
    }
}

for (data_key, model_name_key), wrapper_kwargs in COCO_DATASET_MODEL_WRAPPERS.items():
    wrapper_fn_name = f'get_{data_key}_for_{model_name_key}'
    wrapper_fn = partial(globals()['get_coco_for_ssd'], **wrapper_kwargs)
    globals()[wrapper_fn_name] = wrapper_fn
    DATA_WRAPPER_REGISTRY.register(dataset_name=data_key, model_type=model_name_key)(wrapper_fn)
    __all__.append(wrapper_fn_name)
