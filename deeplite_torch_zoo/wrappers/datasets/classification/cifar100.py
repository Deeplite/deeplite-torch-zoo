import os
from os.path import expanduser

import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import default_collate


__all__ = ["get_cifar100"]


def get_cifar100(
    data_root="", batch_size=128, num_workers=1, download=True, device="cuda", **kwargs
):
    def assign_device(x):
        if device == "cuda":
            return x
        return [v.to(device) for v in x]

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda x: assign_device(default_collate(x)),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda x: assign_device(default_collate(x)),
    )

    return {"train": train_loader, "test": test_loader}
