import os

from ..utils import get_dataloader
from deeplite_torch_zoo.src.objectdetection.ssd.datasets.coco import CocoDetectionBoundingBox


__all__ = ["get_coco_for_fasterrcnn_resnet50_fpn"]


def _get_coco_for_rcnn(data_root, val_ann_file=None, val_dir=None):
    val_annotate = os.path.join(data_root, val_ann_file)
    val_coco_root = os.path.join(data_root, val_dir)
    test_dataset = CocoDetectionBoundingBox(val_coco_root, val_annotate)
    return test_dataset


def get_coco_for_fasterrcnn_resnet50_fpn(
    data_root, batch_size=32, num_workers=4, distributed=False, fp16=False, val_dir="val2017",
    val_ann_file="annotations/instances_val2017.json", device="cuda", **kwargs):

    if len(kwargs):
        import sys
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    test_dataset = _get_coco_for_rcnn(data_root=data_root, val_ann_file=val_ann_file, val_dir=val_dir)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": None, "val": test_loader, "test": test_loader}
