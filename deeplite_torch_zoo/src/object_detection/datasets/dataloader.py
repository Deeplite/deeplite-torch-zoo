# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random

import numpy as np

import torch
from torch.utils.data import dataloader, distributed

from deeplite_torch_zoo.utils import LOGGER, colorstr, de_parallel, torch_distributed_zero_first
from deeplite_torch_zoo.src.object_detection.datasets.dataset import YOLODataset
from deeplite_torch_zoo.src.object_detection.datasets.utils import RANK, PIN_MEMORY


class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers. Uses same syntax as vanilla DataLoader."""

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """Reset iterator.
        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)



def seed_worker(worker_id):  # noqa
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset"""
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0)


def get_dataloader(dataset_path, data, cfg, batch_size=16, rank=0, mode='train'):
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        gs = 32
        # gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(
            cfg,
            dataset_path,
            batch_size,
            data,
            mode=mode,
            rect=mode == 'val',
            stride=gs
        )
    shuffle = mode == 'train'
    if getattr(dataset, 'rect', False) and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    workers = cfg.workers if mode == 'train' else cfg.workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader
