import os

from deeplite_torch_zoo.wrappers.datasets.classification.augs import \
    get_imagenet_transforms
from deeplite_torch_zoo.wrappers.datasets.utils import get_dataloader
from deeplite_torch_zoo.wrappers.registries import DATA_WRAPPER_REGISTRY
from torchvision import datasets

__all__ = ["get_imagenet"]


@DATA_WRAPPER_REGISTRY.register(dataset_name="imagenet16")
@DATA_WRAPPER_REGISTRY.register(dataset_name="imagenet10")
@DATA_WRAPPER_REGISTRY.register(dataset_name="imagenet")
def get_imagenet(data_root, batch_size=128, test_batch_size=None, img_size=224, num_workers=4,
    fp16=False, distributed=False, device="cuda", train_split='imagenet_training',
    val_split='imagenet_val', **kwargs):

    if len(kwargs):
        import sys
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_transforms, val_transforms = get_imagenet_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, train_split),
        train_transforms,
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(data_root, val_split),
        val_transforms,
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_batch_size = batch_size if test_batch_size is None else test_batch_size
    test_loader = get_dataloader(test_dataset, batch_size=test_batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "test": test_loader}
