import os
from torchvision import datasets
from torchvision import transforms

from ..utils import get_dataloader

__all__ = ["get_tinyimagenet"]


def get_tinyimagenet(data_root, batch_size=128, num_workers=4, fp16=False, device="cuda", distributed=False, **kwargs):

    if len(kwargs):
        import sys
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_root, x), data_transforms[x])
                      for x in ['train', 'val']}

    train_loader = get_dataloader(image_datasets["train"], batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(image_datasets["val"], batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "val": test_loader, "test": test_loader}
