import torch
import os

from deeplite_torch_zoo.src.objectdetection.mb_ssd.datasets.coco import CocoDetectionBoundingBox


__all__ = ["get_coco_for_fasterrcnn_resnet50_fpn"]


def _get_coco_for_rcnn(
    data_root,
    val_ann_file=None,
    val_dir=None,
):

    val_annotate = os.path.join(data_root, val_ann_file)
    val_coco_root = os.path.join(data_root, val_dir)
    test_dataset = CocoDetectionBoundingBox(
        val_coco_root,
        val_annotate,
    )
    return test_dataset


def get_coco_loaders_for_rcnn(
    data_root,
    batch_size=32,
    num_workers=4,
    val_ann_file=None,
    val_dir=None,
):
    test_dataset = _get_coco_for_rcnn(
        data_root=data_root,
        val_ann_file=val_ann_file,
        val_dir=val_dir,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return {"train": None, "val": test_loader, "test": test_loader}


def get_coco_for_fasterrcnn_resnet50_fpn(
    data_root,
    batch_size=32,
    **kwargs,
):
    return get_coco_loaders_for_rcnn(
            data_root=data_root,
            batch_size=batch_size,
            val_ann_file="annotations/instances_val2017.json",
            val_dir="val2017",
        )
