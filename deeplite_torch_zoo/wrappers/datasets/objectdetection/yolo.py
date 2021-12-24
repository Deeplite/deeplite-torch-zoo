import os
import sys
from pathlib import Path

from ..utils import get_dataloader

from deeplite_torch_zoo.src.objectdetection.datasets.voc import VocDataset
from deeplite_torch_zoo.src.objectdetection.datasets.voc_utils import prepare_voc_data
from deeplite_torch_zoo.src.objectdetection.datasets.lisa import LISA
from deeplite_torch_zoo.src.objectdetection.datasets.wider_face import WiderFace
from deeplite_torch_zoo.src.objectdetection.datasets.transforms import random_transform_fn
from deeplite_torch_zoo.src.objectdetection.datasets.coco import CocoDetectionBoundingBox

__all__ = [
    "get_coco_for_yolo",
    "get_lisa_for_yolo",
    "get_voc_for_yolo",
    "get_wider_face_for_yolo",
    "get_person_detection_for_yolo",
]


def get_coco_for_yolo(
    data_root, batch_size=32, num_workers=1, num_classes=80, img_size=416,
    fp16=False, distributed=False, device="cuda", **kwargs
):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

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

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=not distributed, collate_fn=train_dataset.collate_img_label_fn, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=False, collate_fn=test_dataset.collate_img_label_fn, device=device)

    return {"train": train_loader, "val": test_loader, "test": test_loader}


def get_lisa_for_yolo(data_root, batch_size=32, num_workers=4, fp16=False, distributed=False, device="cuda", **kwargs):

    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    # setup dataset
    train_dataset = LISA(data_root, _set="train")
    test_dataset = LISA(data_root, _set="valid")

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=not distributed, collate_fn=train_dataset.collate_fn, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=False, collate_fn=test_dataset.collate_fn, device=device)

    return {"train": train_loader, "val": test_loader, "test": test_loader}


def prepare_yolo_voc_data(vockit_data_root, annotation_path, standard_voc_format=True):

    Path(annotation_path).mkdir(parents=True, exist_ok=True)

    train_anno_path = os.path.join(str(annotation_path), "train_annotation.txt")
    test_anno_path = os.path.join(str(annotation_path), "test_annotation.txt")

    if standard_voc_format:
        train_data_paths = [os.path.join(vockit_data_root, "VOC2007"),
            os.path.join(vockit_data_root, "VOC2012")]
        test_data_paths = [os.path.join(vockit_data_root, "VOC2007"),]
    else:
        train_data_paths, test_data_paths = [vockit_data_root,], [vockit_data_root,]

    if not (os.path.exists(train_anno_path) and os.path.exists(test_anno_path)):
        prepare_voc_data(train_data_paths, test_data_paths, annotation_path)


def _get_voc_for_yolo(annotation_path, num_classes=None, img_size=448):
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


def get_voc_for_yolo(
    data_root, batch_size=32, num_workers=4, num_classes=None, img_size=448, fp16=False, distributed=False, device="cuda", **kwargs
):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    annotation_path = os.path.join(data_root, "yolo_data")
    prepare_yolo_voc_data(data_root, annotation_path)
    train_dataset, test_dataset = _get_voc_for_yolo(
        annotation_path, num_classes=num_classes, img_size=img_size
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=not distributed, collate_fn=train_dataset.collate_img_label_fn, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=False, collate_fn=test_dataset.collate_img_label_fn, device=device)


    return {"train": train_loader, "val": test_loader, "test": test_loader}



def _get_widerface_for_yolo(root, num_classes=None, img_size=448):
    train_dataset = WiderFace(
        root=root,
        num_classes=num_classes,
        split="train",
        img_size=img_size,
    )
    test_dataset = WiderFace(
        root=root,
        num_classes=num_classes,
        split="test",
        img_size=img_size,
    )
    return train_dataset, test_dataset


def get_wider_face_for_yolo(
    data_root, batch_size=32, num_workers=4, num_classes=None, img_size=448, fp16=False, distributed=False, device="cuda", **kwargs
):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_dataset, test_dataset = _get_widerface_for_yolo(
        data_root, num_classes=num_classes, img_size=img_size
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=not distributed, collate_fn=train_dataset.collate_img_label_fn, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=False, collate_fn=test_dataset.collate_img_label_fn, device=device)


    return {"train": train_loader, "val": test_loader, "test": test_loader}


def get_person_detection_for_yolo(
    data_root, batch_size=32, num_workers=4, num_classes=1, img_size=448, fp16=False, distributed=False, device="cuda", **kwargs
):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    annotation_path = os.path.join(data_root, "yolo_data")
    prepare_yolo_voc_data(data_root, annotation_path, standard_voc_format=False)
    train_dataset, test_dataset = _get_voc_for_yolo(
        annotation_path, num_classes=num_classes, img_size=img_size
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=not distributed, collate_fn=train_dataset.collate_img_label_fn, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, fp16=fp16,
        distributed=distributed, shuffle=False, collate_fn=test_dataset.collate_img_label_fn, device=device)


    return {"train": train_loader, "val": test_loader, "test": test_loader}
