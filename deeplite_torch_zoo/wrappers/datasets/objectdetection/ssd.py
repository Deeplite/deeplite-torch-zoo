import os
import torch
from torch.utils.data import ConcatDataset

from deeplite_torch_zoo.src.objectdetection.mb_ssd.datasets.voc_dataset import VOCDataset
from deeplite_torch_zoo.src.objectdetection.mb_ssd.datasets.coco import CocoDetectionBoundingBox

from deeplite_torch_zoo.src.objectdetection.mb_ssd.repo.vision.ssd.data_preprocessing import (
    TrainAugmentation,
)
from deeplite_torch_zoo.src.objectdetection.mb_ssd.repo.vision.ssd.ssd import MatchPrior


__all__ = ["get_voc_for_ssd", "get_coco_for_ssd"]


def _get_voc_for_ssd(data_root, config=None, num_classes=21):
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    train_dataset = VOCDataset(
        root=data_root,
        n_classes=num_classes,
        is_test=False,
        transform=train_transform,
        target_transform=target_transform,
    )

    if "VOC2012" not in data_root:
        test_dataset = VOCDataset(root=data_root, n_classes=num_classes, is_test=True)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def get_voc_for_ssd(data_root, config, num_classes=21, batch_size=32, num_workers=4):
    train_dataset_07, test_dataset = _get_voc_for_ssd(
        data_root=os.path.join(data_root, "VOC2007"), config=config, num_classes=num_classes
    )
    train_dataset_12, _ = _get_voc_for_ssd(
        data_root=os.path.join(data_root, "VOC2012"), config=config, num_classes=num_classes
    )
    train_dataset = ConcatDataset([train_dataset_07, train_dataset_12])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return {"train": train_loader, "val": test_loader, "test": test_loader}


def _get_coco_for_ssd(
    data_root,
    config=None,
    img_size=416,
    train_ann_file=None,
    train_dir=None,
    val_ann_file=None,
    val_dir=None,
    missing_ids=[],
    classes=[],
):
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

    train_annotate = os.path.join(data_root, train_ann_file)
    train_coco_root = os.path.join(data_root, train_dir)
    train_dataset = CocoDetectionBoundingBox(
        train_coco_root,
        train_annotate,
        transform=train_transform,
        target_transform=target_transform,
        missing_ids=missing_ids,
        classes=classes,
    )

    val_annotate = os.path.join(data_root, val_ann_file)
    val_coco_root = os.path.join(data_root, val_dir)
    test_dataset = CocoDetectionBoundingBox(
        val_coco_root,
        val_annotate,
        missing_ids=missing_ids,
        classes=classes,
    )

    return train_dataset, test_dataset


def get_coco_for_ssd(
    data_root,
    config,
    batch_size=32,
    num_workers=4,
    train_ann_file=None,
    train_dir=None,
    val_ann_file=None,
    val_dir=None,
    missing_ids=[],
    classes=[],
):
    train_dataset, test_dataset = _get_coco_for_ssd(
        data_root=data_root,
        config=config,
        train_ann_file=train_ann_file,
        train_dir=train_dir,
        val_ann_file=val_ann_file,
        val_dir=val_dir,
        missing_ids=missing_ids,
        classes=classes,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return {"train": train_loader, "val": test_loader, "test": test_loader}
