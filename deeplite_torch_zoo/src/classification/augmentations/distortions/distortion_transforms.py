import numpy as np
from PIL import Image as PILImage
from torchvision import transforms

from deeplite_torch_zoo.src.classification.augmentations.augs import (
    get_vanilla_transforms,
)
from deeplite_torch_zoo.src.classification.augmentations.distortions import (
    DISTORTION_REGISTRY,
)


def generate_distortion_fn(distortion_name, severity):
    aug_fn = DISTORTION_REGISTRY.get(distortion_name)

    def distort_batch(x):
        x = np.asarray(x, dtype=np.uint8)
        x = aug_fn(x, severity=severity)
        return PILImage.fromarray(np.uint8(x))

    return distort_batch


def get_distortion_transforms(distortion_name, img_size, severity=1, **kwargs):
    distortion_transform = transforms.Lambda(
        generate_distortion_fn(distortion_name, severity=severity)
    )
    train_transforms, val_transforms = get_vanilla_transforms(
        img_size,
        add_train_transforms=distortion_transform,
        add_test_transforms=distortion_transform,
        **kwargs
    )
    return train_transforms, val_transforms
