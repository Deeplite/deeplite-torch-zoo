import os
import torchvision

from os.path import expanduser
from ..utils import get_dataloader
from torchvision import transforms


__all__ = ["get_cifar100"]


def get_cifar100(
    data_root="", batch_size=128, num_workers=4, fp16=False, download=True, device="cuda", distributed=False, **kwargs
):
    if len(kwargs):
        import sys
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=True,
        download=download,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        download=download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "test": test_loader}
