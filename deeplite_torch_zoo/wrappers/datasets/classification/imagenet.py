import os

import torch
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate


__all__ = ["get_imagenet"]


def get_imagenet(data_root="", train_dir="imagenet_training", val_dir="imagenet_val", batch_size=128, num_workers=4, device="cuda", **kwargs):
    def assign_device(x):
        #if device == "cuda":
        #    return x
        return [v.to(device) for v in x]

    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, train_dir),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
            ]
        ),
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(data_root, val_dir),
        transforms.Compose(
            [
                transforms.Resize(64),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        #pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda x: assign_device(default_collate(x)),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        #pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda x: assign_device(default_collate(x)),
    )

    return {"train": train_loader, "test": test_loader}
