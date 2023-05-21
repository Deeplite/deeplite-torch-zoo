# Code modified from: https://github.com/DeepVoltaire/AutoAugment

import numpy as np
import torch
from timm.data.transforms_factory import (
    transforms_imagenet_eval,
    transforms_imagenet_train,
)
from torchvision import transforms

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_imagenet_transforms(
    img_size,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    crop_pct=DEFAULT_CROP_PCT,
    scale=(0.08, 1.0),  # Random resize scale
    ratio=(3.0 / 4.0, 4.0 / 3.0),  # Random resize aspect ratio
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,  # Color jitter factor
    auto_augment=None,  # Use AutoAugment policy. Choices: 'v0', 'v0r', 'original', 'originalr', 'randaugment', 'augmix'
    train_interpolation='random',  # Training interpolation (random, bilinear, bicubic default: "random")
    test_interpolation='bilinear',
    re_prob=0.0,  # Random erase prob (default: 0.)
    re_mode='pixel',  # Random erase mode (default: "pixel")
    re_count=1,  # Random erase count (default: 1)
):
    train_transforms = transforms_imagenet_train(
        img_size=img_size,
        mean=mean,
        std=std,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=train_interpolation,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
    )
    val_transforms = transforms_imagenet_eval(
        img_size,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        interpolation=test_interpolation,
    )
    return train_transforms, val_transforms


def get_vanilla_transforms(
    img_size,
    hflip=0.5,
    jitter=0.4,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    crop_pct=DEFAULT_CROP_PCT,
    add_train_transforms=None,
    add_test_transforms=None,
    cutout_args=None,
):
    train_transforms = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(hflip),
        transforms.ColorJitter(
            brightness=jitter, contrast=jitter, saturation=jitter, hue=0
        ),
    ]
    if add_train_transforms is not None:
        train_transforms.append(add_train_transforms)

    train_transforms.append(transforms.ToTensor())

    if cutout_args is not None:
        train_transforms.append(Cutout(**cutout_args))

    train_transforms.append(transforms.Normalize(mean, std))

    test_transforms = [
        transforms.Resize(int(img_size / crop_pct)),
        transforms.CenterCrop(img_size),
    ]
    if add_test_transforms is not None:
        test_transforms.append(add_test_transforms)

    test_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
