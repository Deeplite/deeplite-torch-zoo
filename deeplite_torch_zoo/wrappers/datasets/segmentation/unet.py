import sys
import os

import albumentations as albu
from ..utils import get_dataloader
from deeplite_torch_zoo.src.segmentation.datasets.pascal_voc import PascalVocDataset
from deeplite_torch_zoo.src.segmentation.datasets.carvana import BasicDataset


__all__ = ["get_carvana_for_unet", "get_voc_for_unet"]


def get_carvana_for_unet(data_root, batch_size=4, num_workers=4, fp16=False, distributed=False, device="cuda", **kwargs):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_dataset = BasicDataset(
        os.path.join(data_root, "train_imgs/"),
        os.path.join(data_root, "train_masks/"),
        scale=0.5,
    )
    test_dataset = BasicDataset(
        os.path.join(data_root, "val_imgs/"),
        os.path.join(data_root, "val_masks/"),
        scale=0.5,
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "test": test_loader}


def get_voc_for_unet(
    data_root, num_classes=21, batch_size=4, num_workers=4, img_size=512, net="unet", fp16=False, distributed=False, device="cuda", **kwargs
):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    # Dataset
    affine_augmenter = albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            # Rotate(5, p=.5)
        ]
    )
    image_augmenter = albu.Compose([albu.GaussNoise(p=0.5), albu.RandomBrightnessContrast(p=0.5)])

    train_dataset = PascalVocDataset(
        base_dir=data_root,
        num_classes=num_classes,
        split="train_aug",
        affine_augmenter=affine_augmenter,
        image_augmenter=image_augmenter,
        target_size=(img_size, img_size),
        net_type=net,
    )
    test_dataset = PascalVocDataset(
        base_dir=data_root,
        num_classes=num_classes,
        split="valid",
        target_size=(img_size, img_size),
        net_type=net,
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "test": test_loader}
