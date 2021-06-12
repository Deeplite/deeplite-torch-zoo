import os
import albumentations as albu
from torch.utils.data import DataLoader
from deeplite_torch_zoo.src.segmentation.datasets.pascal_voc import PascalVocDataset
from deeplite_torch_zoo.src.segmentation.datasets.carvana import BasicDataset


__all__ = ["get_carvana_for_unet", "get_voc_for_unet"]


def get_carvana_for_unet(data_root, batch_size=4, num_workers=4, **kwargs):
    train_dataset = BasicDataset(
        os.path.join(data_root, "train_imgs/"),
        os.path.join(data_root, "train_masks/"),
        scale=0.5,
    )
    valid_dataset = BasicDataset(
        os.path.join(data_root, "val_imgs/"),
        os.path.join(data_root, "val_masks/"),
        scale=0.5,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return {"train": train_loader, "test": val_loader}


def get_voc_for_unet(
    data_root, num_classes=21, batch_size=4, num_workers=4, img_size=512, net="unet", **kwargs
):
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
    valid_dataset = PascalVocDataset(
        base_dir=data_root,
        num_classes=num_classes,
        split="valid",
        target_size=(img_size, img_size),
        net_type=net,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "test": val_loader}
