import os
from os.path import expanduser

import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler as DS

__all__ = ["get_mnist"]


def get_mnist(data_root="", batch_size=128, num_workers=4, fp16=False, download=True, device="cuda", distributed=False, **kwargs):

    if len(kwargs):
        import sys
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    def half_precision(x):
        if fp16:
            x = [_x.half() if isinstance(_x, torch.FloatTensor) else _x for _x in x]
        return x

    def assign_device(x):
        if x[0].is_cuda ^ (device == "cuda"):
            return x
        return [v.to(device) for v in x]

    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")

    train_dataset = torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        download=download,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_root,
        train=False,
        download=download,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not distributed,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda x: half_precision(assign_device(default_collate(x))),
        sampler=DS(train_dataset) if distributed else None,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda x: half_precision(assign_device(default_collate(x))),
        sampler=DS(test_dataset) if distributed else None,
    )

    return {"train": train_loader, "test": test_loader}
