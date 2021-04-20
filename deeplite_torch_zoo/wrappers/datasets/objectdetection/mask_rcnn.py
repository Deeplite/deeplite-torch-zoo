import torch
import os

from deeplite_torch_zoo.src.objectdetection.mb_ssd.datasets.coco import CocoDetectionBoundingBox
from deeplite_torch_zoo.src.objectdetection.datasets.transforms import TensorizeTransform


__all__ = ["get_coco_for_fasterrcnn_resnet50_fpn", "get_coco_for_maskrcnn_resnet50_fpn"]


def _get_coco_for_rcnn(
    data_root,
    ann_file=None,
    set_dir=None,
):

    ann_path = os.path.join(data_root, ann_file)
    set_coco_root = os.path.join(data_root, set_dir)
    dataset = CocoDetectionBoundingBox(
        set_coco_root,
        ann_path,
        transform=TensorizeTransform()
    )
    return dataset


def collate_tv(data):
    images, boxes, labels, ids = [], [], [], []
    for im, box, label, _id in data:
        images.append(im)
        boxes.append(torch.tensor(box))
        labels.append(torch.tensor(label))
        ids.append(torch.tensor(_id))
    return images, boxes, labels, ids


def get_coco_loaders_for_rcnn(
    data_root,
    batch_size=32,
    num_workers=4,
    val_ann_file=None,
    val_dir=None,
    train_ann_file=None,
    train_dir=None,
):
    train_dataset = _get_coco_for_rcnn(
        data_root=data_root,
        ann_file=train_ann_file,
        set_dir=train_dir,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_tv,
    )

    test_dataset = _get_coco_for_rcnn(
        data_root=data_root,
        ann_file=val_ann_file,
        set_dir=val_dir,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_tv,
    )
    return {"train": train_loader, "val": test_loader, "test": test_loader}


def get_coco_for_fasterrcnn_resnet50_fpn(
    data_root,
    batch_size=32,
    num_workers=4,
    **kwargs,
):
    return get_coco_loaders_for_rcnn(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_ann_file="annotations/instances_val2017.json",
            val_dir="val2017",
            train_ann_file="annotations/instances_train2017.json",
            train_dir="train2017",
        )
def get_coco_for_maskrcnn_resnet50_fpn(
    data_root,
    batch_size=32,
    num_workers=4,
    **kwargs,
):
    return get_coco_for_fasterrcnn_resnet50_fpn(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers
    )

