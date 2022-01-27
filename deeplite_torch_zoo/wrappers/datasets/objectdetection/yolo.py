import os
import sys
from collections import namedtuple

from deeplite_torch_zoo.wrappers.datasets.utils import get_dataloader
from deeplite_torch_zoo.src.objectdetection.datasets.voc import VocDataset
from deeplite_torch_zoo.src.objectdetection.datasets.voc_utils import prepare_yolo_voc_data
from deeplite_torch_zoo.src.objectdetection.datasets.lisa import LISA
from deeplite_torch_zoo.src.objectdetection.datasets.wider_face import WiderFace
from deeplite_torch_zoo.src.objectdetection.datasets.transforms import random_transform_fn
from deeplite_torch_zoo.src.objectdetection.datasets.coco import CocoDetectionBoundingBox


__all__ = []


def make_dataset_wrapper(wrapper_name, num_classes, img_size, dataset_create_fn):
    def wrapper_func(data_root, batch_size=32, num_workers=1, num_classes=num_classes,
        img_size=img_size, fp16=False, distributed=False, device="cuda", **kwargs):

        if len(kwargs):
            print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

        train_dataset, test_dataset = dataset_create_fn(data_root, num_classes, img_size)

        train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
            fp16=fp16, distributed=distributed, shuffle=not distributed,
            collate_fn=train_dataset.collate_img_label_fn, device=device)

        test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
            distributed=distributed, shuffle=False, collate_fn=test_dataset.collate_img_label_fn, device=device)

        return {"train": train_loader, "val": test_loader, "test": test_loader}

    wrapper_func.__name__ = wrapper_name
    return wrapper_func


def create_coco_datasets(data_root, num_classes, img_size):

    from deeplite_torch_zoo.src.objectdetection.configs.coco_config import MISSING_IDS, DATA

    train_trans = random_transform_fn
    train_annotate = os.path.join(data_root, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(data_root, "train2017")
    train_dataset = CocoDetectionBoundingBox(
        train_coco_root, train_annotate, num_classes=num_classes, transform=train_trans,
        img_size=img_size, classes=DATA["CLASSES"], missing_ids=MISSING_IDS
    )

    val_annotate = os.path.join(data_root, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(data_root, "val2017")
    test_dataset = CocoDetectionBoundingBox(
        val_coco_root, val_annotate, num_classes=num_classes, img_size=img_size,
        classes=DATA["CLASSES"], missing_ids=MISSING_IDS
    )

    return train_dataset, test_dataset


def create_lisa_datasets(data_root, num_classes, img_size):
    return LISA(data_root, _set="train", img_size=img_size), LISA(data_root, _set="valid", img_size=img_size)


def create_voc_datasets(data_root, num_classes, img_size, is_07_subset=False, standard_voc_format=True):
    annotation_path = os.path.join(data_root, "yolo_data")
    prepare_yolo_voc_data(data_root, annotation_path,
        is_07_subset=is_07_subset, standard_voc_format=standard_voc_format)
    train_dataset = VocDataset(
        num_classes=num_classes,
        annotation_path=annotation_path,
        anno_file_type="train",
        img_size=img_size,
    )
    test_dataset = VocDataset(
        num_classes=num_classes,
        annotation_path=annotation_path,
        anno_file_type="test",
        img_size=img_size,
    )
    return train_dataset, test_dataset


def create_widerface_datasets(data_root, num_classes, img_size):
    train_dataset = WiderFace(
        root=data_root,
        num_classes=num_classes,
        split="train",
        img_size=img_size,
    )
    test_dataset = WiderFace(
        root=data_root,
        num_classes=num_classes,
        split="test",
        img_size=img_size,
    )
    return train_dataset, test_dataset


def create_voc07_datasets(data_root, num_classes, img_size):
    return create_voc_datasets(data_root, num_classes, img_size, is_07_subset=True)


def create_person_detection_datasets(data_root, num_classes, img_size):
    return create_voc_datasets(data_root, num_classes, img_size, standard_voc_format=False)


DatasetParameters = namedtuple('DatasetParameters', ['num_classes', 'img_size', 'dataset_create_fn'])
DATASET_WRAPPER_FNS = {
    'coco': DatasetParameters(80, 416, create_coco_datasets),
    'lisa': DatasetParameters(20, 416, create_lisa_datasets),
    'voc': DatasetParameters(20, 448, create_voc_datasets),
    'voc07': DatasetParameters(20, 448, create_voc07_datasets),
    'widerface': DatasetParameters(1, 448, create_widerface_datasets),
    'person_detection': DatasetParameters(1, 320, create_person_detection_datasets),
}

for dataset_name, dataset_parameters in DATASET_WRAPPER_FNS.items():
    wrapper_name = f'get_{dataset_name}_for_yolo'
    globals()[wrapper_name] = make_dataset_wrapper(wrapper_name, num_classes=dataset_parameters.num_classes,
        img_size=dataset_parameters.img_size, dataset_create_fn=dataset_parameters.dataset_create_fn)
    __all__.append(wrapper_name)
