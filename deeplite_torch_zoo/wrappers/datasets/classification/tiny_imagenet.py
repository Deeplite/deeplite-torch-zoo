import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data as data
from torch.utils.data.dataloader import default_collate


__all__ = ["get_tinyimagenet"]


def get_tinyimagenet(data_root, batch_size=128, num_workers=4, device="cuda"):
    def assign_device(x):
        return [v.to(device) for v in x]

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
    dataloaders = {
            x: data.DataLoader(
                image_datasets[x],
                batch_size=batch_size,
                shuffle=(x=="train"),
                num_workers=num_workers,
                collate_fn=lambda x: assign_device(default_collate(x))
            ) for x in ['train', 'val']
        }
    dataloaders["test"] = dataloaders["val"]
    return dataloaders
