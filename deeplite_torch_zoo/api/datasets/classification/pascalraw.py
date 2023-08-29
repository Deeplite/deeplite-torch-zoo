import os
from os.path import expanduser
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import torch

from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY
from deeplite_torch_zoo.src.classification.datasets.pascalraw_multi_label_cls import PASCALRAWMultiLabelCls

__all__ = ["get_pascal_raw"]


@DATASET_WRAPPER_REGISTRY.register(dataset_name="pascal_raw")
def get_pascal_raw(
    data_root="",
    batch_size=2,
    test_batch_size=None,
    num_workers=4,
    fp16=False,
    distributed=False,
    train_transforms=None,
    val_transforms=None,
    input_size=[224, 224, 3],
    **kwargs,
):
    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")
    image_folder = data_root
    train_dataset = PASCALRAWMultiLabelCls(root=image_folder, split='train', transform=train_transforms)
    test_dataset = PASCALRAWMultiLabelCls(root=image_folder, split='test', transform=val_transforms)

    train_loader = create_loader(
        train_dataset,
        input_size=input_size,
        batch_size=batch_size,
        is_training=True,
        num_workers=num_workers,
        distributed=distributed,
        img_dtype=torch.float16,
    )

    test_batch_size = batch_size if test_batch_size is None else test_batch_size
    test_loader = create_loader(
        test_dataset,
        input_size=input_size,
        batch_size=test_batch_size,
        is_training=False,
        num_workers=num_workers,
        distributed=distributed,
        img_dtype=torch.float16,
    )

    return {"train": train_loader, "test": test_loader}