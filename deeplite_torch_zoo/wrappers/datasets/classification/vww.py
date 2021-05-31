import os

import pyvww
import torch
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler as DS


__all__ = ["get_vww"]


def get_vww(data_root="", batch_size=128, num_workers=0, device="cuda", distributed=False, **kwargs):

    def assign_device(x, device="cuda"):
        return [v.to(device) for v in x]

    train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=os.path.join(data_root, "all"),
        annFile=os.path.join(data_root, "annotations/instances_train.json"),
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )

    test_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=os.path.join(data_root, "all"),
        annFile=os.path.join(data_root, "annotations/instances_val.json"),
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False if distributed else True,
        num_workers=num_workers,
        collate_fn=lambda x: assign_device(default_collate(x), device=device),
        sampler=DS(train_dataset) if distributed else None,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: assign_device(default_collate(x)),
        sampler=DS(test_dataset) if distributed else None,
    )

    return {"train": train_loader, "test": test_loader}
