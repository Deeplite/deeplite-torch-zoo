import os
import sys
from torch.utils.data import ConcatDataset

from ..utils import get_dataloader
from deeplite_torch_zoo.src.objectdetection.ssd.datasets.voc_dataset import VOCDataset
from deeplite_torch_zoo.src.objectdetection.ssd.datasets.coco import CocoDetectionBoundingBox
from deeplite_torch_zoo.src.objectdetection.ssd.datasets.wider_face import WiderFace
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.data_preprocessing import TrainAugmentation
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.ssd import MatchPrior


__all__ = ["get_voc_for_ssd", "get_coco_for_ssd", "get_wider_face_for_ssd"]


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


def get_voc_for_ssd(data_root, config, num_classes=21, batch_size=32, num_workers=4,
    fp16=False, distributed=False, device="cuda", **kwargs):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_dataset_07, test_dataset = _get_voc_for_ssd(
        data_root=os.path.join(data_root, "VOC2007"), config=config, num_classes=num_classes
    )
    train_dataset_12, _ = _get_voc_for_ssd(
        data_root=os.path.join(data_root, "VOC2012"), config=config, num_classes=num_classes
    )
    train_dataset = ConcatDataset([train_dataset_07, train_dataset_12])

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "val": test_loader, "test": test_loader}


def _get_coco_for_ssd(data_root, config=None, img_size=416, train_ann_file=None, train_dir=None,
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


def get_coco_for_ssd(data_root, config, batch_size=32, num_workers=4, train_ann_file=None, train_dir=None,
    val_ann_file=None, val_dir=None, missing_ids=[], classes=[], fp16=False, distributed=False, device="cuda", **kwargs):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_dataset, test_dataset = _get_coco_for_ssd(data_root=data_root, config=config, train_ann_file=train_ann_file,
        train_dir=train_dir, val_ann_file=val_ann_file, val_dir=val_dir, missing_ids=missing_ids, classes=classes)

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "val": test_loader, "test": test_loader}


def _get_wider_face_for_ssd(data_root, config=None):
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

    train_dataset = WiderFace(root=data_root, split="train",
        transform=train_transform, target_transform=target_transform)
    test_dataset = WiderFace(root=data_root, split="val")

    return train_dataset, test_dataset


def get_wider_face_for_ssd(data_root, config, batch_size=32, num_workers=4,
    fp16=False, distributed=False, device="cuda", **kwargs):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_dataset, test_dataset = _get_wider_face_for_ssd(
        data_root=os.path.join(data_root), config=config
    )
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "val": test_loader, "test": test_loader}
