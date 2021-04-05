import os
import torch
from pathlib import Path

from deeplite_torch_zoo.src.objectdetection.yolov3.utils import VocDataset
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.voc import prepare_data
from deeplite_torch_zoo.src.objectdetection.datasets.lisa import LISA
from deeplite_torch_zoo.src.objectdetection.datasets.image import ImageFolder
from deeplite_torch_zoo.src.objectdetection.datasets.transforms import random_transform_fn
from deeplite_torch_zoo.src.objectdetection.datasets.coco import CocoDetectionBoundingBox

__all__ = [
    "get_coco_for_yolo",
    "get_image_to_folder_for_yolo",
    "get_lisa_for_yolo",
    "get_voc_for_yolo",
]


def get_coco_for_yolo(
    data_root, batch_size=32, num_workers=1, num_classes=80, img_size=416, **kwargs
):
    train_trans = random_transform_fn
    train_annotate = os.path.join(data_root, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(data_root, "images/train2017")
    train_coco = CocoDetectionBoundingBox(
        train_coco_root,
        train_annotate,
        num_classes=num_classes,
        transform=train_trans,
        img_size=img_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_coco,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_coco.collate_img_label_fn,
    )

    val_annotate = os.path.join(data_root, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(data_root, "images/val2017")
    val_coco = CocoDetectionBoundingBox(
        val_coco_root, val_annotate, num_classes=num_classes, img_size=img_size
    )

    val_loader = torch.utils.data.DataLoader(
        val_coco,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=val_coco.collate_img_label_fn,
    )

    return {"train": train_loader, "val": val_loader}


def get_image_to_folder_for_yolo(data_root, batch_size=128, num_workers=4, **kwargs):
    dataset = ImageFolder(data_root)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    data_splits = {"test": data_loader}
    return data_splits


def get_lisa_for_yolo(data_root, batch_size=32, num_workers=4, **kwargs):

    # setup dataset
    train_dataset = LISA(data_root, _set="train")
    val_dataset = LISA(data_root, _set="valid")

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
    )

    return {"train": train_loader, "val": val_dataloader, "test": val_dataloader}


def prepare_yolo_data(vockit_data_root, annotation_path):

    Path(annotation_path).mkdir(parents=True, exist_ok=True)

    train_anno_path = os.path.join(str(annotation_path), "train_annotation.txt")
    test_anno_path = os.path.join(str(annotation_path), "test_annotation.txt")
    if not (os.path.exists(train_anno_path) and os.path.exists(test_anno_path)):
        prepare_data(vockit_data_root, annotation_path)


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
    data_root, batch_size=32, num_workers=4, num_classes=None, img_size=448, device="cuda", **kwargs
):
    def assign_device(x):
        if device == "cuda":
            return x
        return [v.to(device) for v in x]

    annotation_path = os.path.join(data_root, "yolo_data")
    prepare_yolo_data(data_root, annotation_path)
    train_dataset, test_dataset = _get_voc_for_yolo(
        annotation_path, num_classes=num_classes, img_size=img_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda x: assign_device(train_dataset.collate_img_label_fn(x)),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda x: assign_device(test_dataset.collate_img_label_fn(x)),
    )
    return {"train": train_loader, "val": test_loader, "test": test_loader}
