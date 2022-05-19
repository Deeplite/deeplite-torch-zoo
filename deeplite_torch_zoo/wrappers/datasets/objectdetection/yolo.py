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
from deeplite_torch_zoo.src.objectdetection.datasets.coco_config import COCO_MISSING_IDS, COCO_DATA_CATEGORIES
from deeplite_torch_zoo.wrappers.registries import DATA_WRAPPER_REGISTRY


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


def create_coco_datasets(data_root, num_classes, img_size, subsample_categories=None):

    train_trans = random_transform_fn
    train_annotate = os.path.join(data_root, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(data_root, "train2017")


    if subsample_categories is not None:
        categories = subsample_categories
        category_indices = [COCO_DATA_CATEGORIES["CLASSES"].index(cat) + 1 for cat in categories]
        missing_ids = [category for category in list(range(1, 92)) if category not in category_indices]
    else:
        categories = COCO_DATA_CATEGORIES["CLASSES"]
        category_indices = 'all'
        missing_ids = COCO_MISSING_IDS

    train_dataset = CocoDetectionBoundingBox(
        train_coco_root, train_annotate, num_classes=num_classes, transform=train_trans,
        img_size=img_size, classes=categories, category=category_indices, missing_ids=missing_ids
    )

    val_annotate = os.path.join(data_root, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(data_root, "val2017")
    test_dataset = CocoDetectionBoundingBox(
        val_coco_root, val_annotate, num_classes=num_classes, img_size=img_size,
        classes=categories, category=category_indices, missing_ids=missing_ids
    )

    return train_dataset, test_dataset


def create_lisa_datasets(data_root, num_classes, img_size):
    return LISA(data_root, _set="train", img_size=img_size), LISA(data_root, _set="valid", img_size=img_size)


def create_voc_datasets(data_root, num_classes, img_size, is_07_subset=False, standard_voc_format=True,
    class_names=None):
    annotation_path = os.path.join(data_root, "yolo_data")
    prepare_yolo_voc_data(data_root, annotation_path,
        is_07_subset=is_07_subset, standard_voc_format=standard_voc_format)
    train_dataset = VocDataset(
        annotation_path=annotation_path,
        anno_file_type="train",
        img_size=img_size,
        class_names=class_names,
    )
    test_dataset = VocDataset(
        annotation_path=annotation_path,
        anno_file_type="test",
        img_size=img_size,
        class_names=class_names
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
    """VOC2007 dataset with a 'train' set for training and 'val' set for testing"""
    return create_voc_datasets(data_root, num_classes, img_size, is_07_subset=True)


def create_voc_format_datasets(data_root, num_classes, img_size):
    return create_voc_datasets(data_root, num_classes, img_size, standard_voc_format=False)


def create_car_detection_datasets(data_root, num_classes, img_size):
    """Part of COCO containing only the 'car' class"""
    return create_coco_datasets(data_root, num_classes, img_size, subsample_categories=['car'])


def create_person_detection_datasets(data_root, num_classes, img_size):
    """Person detection (1 class) dataset in VOC format"""
    return create_voc_datasets(data_root, num_classes, img_size, standard_voc_format=False, class_names=['person'])


DatasetParameters = namedtuple('DatasetParameters', ['num_classes', 'img_size', 'dataset_create_fn'])
DATASET_WRAPPER_FNS = {
    'coco': DatasetParameters(80, 416, create_coco_datasets),
    'lisa': DatasetParameters(20, 416, create_lisa_datasets),
    'voc': DatasetParameters(20, 448, create_voc_datasets),
    'voc07': DatasetParameters(20, 448, create_voc07_datasets),
    'voc_format_dataset': DatasetParameters(1, 320, create_voc_format_datasets),
    'person_detection': DatasetParameters(1, 320, create_person_detection_datasets),
    'wider_face': DatasetParameters(1, 448, create_widerface_datasets),
    'car_detection': DatasetParameters(1, 320, create_car_detection_datasets),
}

for dataset_name_key, dataset_parameters in DATASET_WRAPPER_FNS.items():
    wrapper_fn_name = f'get_{dataset_name_key}_for_yolo'
    wrapper_fn = make_dataset_wrapper(wrapper_fn_name, num_classes=dataset_parameters.num_classes,
        img_size=dataset_parameters.img_size, dataset_create_fn=dataset_parameters.dataset_create_fn)
    globals()[wrapper_fn_name] = wrapper_fn
    DATA_WRAPPER_REGISTRY.register(dataset_name=dataset_name_key, model_type='yolo')(wrapper_fn)
    __all__.append(wrapper_fn_name)
