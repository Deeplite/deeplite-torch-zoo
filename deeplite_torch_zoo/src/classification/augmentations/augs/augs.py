from timm.data.transforms_factory import (
    transforms_imagenet_eval,
    transforms_imagenet_train,
)
from timm.data.transforms import ToNumpy
from torchvision import transforms

from deeplite_torch_zoo.src.classification.augmentations.augs.cutout import Cutout


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
    re_num_splits=0,
    use_prefetcher=False,
):
    if isinstance(img_size, (tuple, list)):
        img_size = img_size[-2:]
    else:
        img_size = img_size
    train_transforms = transforms_imagenet_train(
        img_size,
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
        re_num_splits=re_num_splits,
        use_prefetcher=use_prefetcher,
    )
    val_transforms = transforms_imagenet_eval(
        img_size,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        interpolation=test_interpolation,
        use_prefetcher=use_prefetcher,
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
    use_prefetcher=False,
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

    if not use_prefetcher:
        train_transforms.append(transforms.ToTensor())
    else:
        train_transforms.append(ToNumpy())

    if cutout_args is not None:
        train_transforms.append(Cutout(**cutout_args))

    if not use_prefetcher:
        train_transforms.append(transforms.Normalize(mean, std))

    test_transforms = [
        transforms.Resize(int(img_size / crop_pct)),
        transforms.CenterCrop(img_size),
    ]
    if add_test_transforms is not None:
        test_transforms.append(add_test_transforms)

    if not use_prefetcher:
        test_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    else:
        test_transforms.append(ToNumpy())

    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)
