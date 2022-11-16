import os

from deeplite_torch_zoo.wrappers.datasets.classification.augs import \
    get_vanilla_transforms
from deeplite_torch_zoo.wrappers.datasets.utils import get_dataloader
from deeplite_torch_zoo.wrappers.registries import DATA_WRAPPER_REGISTRY
from torchvision import datasets

__all__ = ["get_tinyimagenet"]


@DATA_WRAPPER_REGISTRY.register(dataset_name='tinyimagenet')
def get_tinyimagenet(data_root, batch_size=128, num_workers=4, fp16=False, img_size=64,
    device="cuda", distributed=False, **kwargs):

    if len(kwargs):
        import sys
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_transforms, val_transforms = get_vanilla_transforms(
        img_size,
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2302, 0.2265, 0.2262),
        crop_pct = 1.0,
    )

    data_transforms = {'train': train_transforms, 'val': val_transforms}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_root, x), data_transforms[x])
                      for x in ['train', 'val']}

    train_loader = get_dataloader(image_datasets["train"], batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(image_datasets["val"], batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "val": test_loader, "test": test_loader}
