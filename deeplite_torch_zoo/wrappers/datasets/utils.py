import torch

from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler as DS


def get_dataloader(
    dataset, batch_size=32, num_workers=4, fp16=False, distributed=False, shuffle=False,
    collate_fn=None, device="cuda"
):
    if collate_fn is None:
        collate_fn = default_collate
    def half_precision(x):
        x = collate_fn(x)
        x = [_x.half() if isinstance(_x, torch.FloatTensor) else _x for _x in x]
        return x

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn= half_precision if fp16 else collate_fn,
        sampler=DS(dataset) if distributed else None,
    )
    return dataloader
