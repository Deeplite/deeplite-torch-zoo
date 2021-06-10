import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler as DS


__all__ = ["get_tinyimagenet"]


def get_tinyimagenet(data_root, batch_size=128, num_workers=4, fp16=False, device="cuda", distributed=False, **kwargs):

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
                shuffle=False if distributed else (x=="train"),
                num_workers=0,
                collate_fn=lambda x: half_precision(assign_device(default_collate(x))),
                sampler=DS(image_datasets[x]) if distributed else None
            ) for x in ['train', 'val']
        }
    dataloaders["test"] = dataloaders["val"]
    return dataloaders
