import torch

from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler as DS

import torch
from pathlib import Path
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
import numpy as np
from typing import List

def get_dataloader(
    dataset,
    batch_size=32,
    num_workers=4,
    fp16=False,
    distributed=False,
    shuffle=False,
    collate_fn=None,
    device="cuda",
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
        collate_fn=half_precision if fp16 else collate_fn,
        sampler=DS(dataset) if distributed else None,
    )
    return dataloader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def create_train_loader(train_dataset, num_workers, batch_size,
                            distributed, in_memory, img_size):
        this_device = f'cuda:{0}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = img_size
        decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToTorchImage(),
            ToDevice(torch.device(this_device), non_blocking=True),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

def create_val_loader(val_dataset, num_workers, batch_size,
                          img_size, distributed):
        this_device = f'cuda:{0}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (img_size, img_size)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader
