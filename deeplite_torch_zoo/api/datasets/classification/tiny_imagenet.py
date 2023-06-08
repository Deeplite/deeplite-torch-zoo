import os

from torchvision import datasets

from deeplite_torch_zoo.src.classification.augmentations.augs import (
    get_vanilla_transforms,
)
from deeplite_torch_zoo.api.datasets.utils import get_dataloader
from deeplite_torch_zoo.api.registries import DATA_WRAPPER_REGISTRY

__all__ = ["get_tinyimagenet"]


@DATA_WRAPPER_REGISTRY.register(dataset_name='tinyimagenet')
def get_tinyimagenet(
    data_root,
    batch_size=128,
    test_batch_size=None,
    num_workers=4,
    fp16=False,
    img_size=64,
    distributed=False,
    train_transforms=None,
    val_transforms=None,
    **kwargs,
):
    if len(kwargs):
        import sys

        print(
            f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!"
        )

    default_train_transforms, default_val_transforms = get_vanilla_transforms(
        img_size,
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2302, 0.2265, 0.2262),
        crop_pct=1.0,
    )

    train_transforms = (
        train_transforms if train_transforms is not None else default_train_transforms
    )
    val_transforms = (
        val_transforms if val_transforms is not None else default_val_transforms
    )

    data_transforms = {'train': train_transforms, 'val': val_transforms}
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_root, x), data_transforms[x])
        for x in ['train', 'val']
    }

    train_loader = get_dataloader(
        image_datasets["train"],
        batch_size=batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=not distributed,
    )

    test_batch_size = batch_size if test_batch_size is None else test_batch_size
    test_loader = get_dataloader(
        image_datasets["val"],
        batch_size=test_batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=False,
    )

    return {"train": train_loader, "val": test_loader, "test": test_loader}
