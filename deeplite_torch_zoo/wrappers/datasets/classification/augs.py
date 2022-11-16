import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.transforms_factory import (transforms_imagenet_eval,
                                          transforms_imagenet_train)

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_imagenet_transforms(
    img_size,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    crop_pct=DEFAULT_CROP_PCT,
    scale=(0.08, 1.0),  # Random resize scale
    ratio=(3./4., 4./3.),  # Random resize aspect ratio
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
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    crop_pct=DEFAULT_CROP_PCT,
    auto_aug=False,
):
    T_train = [A.RandomResizedCrop(height=img_size, width=img_size, scale=scale, ratio=ratio)]
    if auto_aug:
        pass
        # TODO: implement AugMix, AutoAug & RandAug in albumentation
    else:
        if hflip > 0:
            T_train += [A.HorizontalFlip(p=hflip)]
        if vflip > 0:
            T_train += [A.VerticalFlip(p=vflip)]
        if jitter > 0:
            color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, satuaration, 0 hue
            T_train += [A.ColorJitter(*color_jitter, 0)]

    # Use fixed crop for eval set (reproducibility)
    T_test = [A.SmallestMaxSize(max_size=int(img_size / crop_pct)), A.CenterCrop(height=img_size, width=img_size)]

    T_train += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
    T_test += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor

    return A.Compose(T_train), A.Compose(T_test)
