import os
from os.path import expanduser
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import rawpy

from deeplite_torch_zoo.api.datasets.utils import get_dataloader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY


__all__ = ["get_pascal_raw"]


@DATASET_WRAPPER_REGISTRY.register(dataset_name="raw_pascal_raw")
def get_pascal_raw(
    data_root="",
    batch_size=2,
    test_batch_size=None,
    num_workers=4,
    fp16=False,
    distributed=False,
    train_transforms=None,
    val_transforms=None,
    **kwargs,
):
    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")
    image_folder = os.path.join(data_root, "NEF")
    train_dataset = PASCALRAWDataset(
        root=image_folder, metadata_file=metadata_file,
        split='train', transform=train_transforms)

    test_dataset = PASCALRAWDataset(
        root=image_folder, metadata_file=metadata_file,
        split='test', transform=val_transforms)

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=not distributed,
    )

    test_batch_size = batch_size if test_batch_size is None else test_batch_size
    test_loader = get_dataloader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        shuffle=False,
    )

    return {"train": train_loader, "test": test_loader}

class PASCALRAWDataset(Dataset):
    def __init__(self,
                 root,
                 split: str = "train",
                 transform = None,
                 split_percentage=0.8,
                 ):
        '''
        root: Assuming the tiff files are grouped into folders of corresponding class,
              the root director should point to the parent folder of these folders.
        '''
        self.data_dir = root
        self.transform = transform
        self.split = split
        self.image_files = glob.glob(os.path.join(root, "*.NEF"))
        self.image_files = glob.glob(os.path.join(root, "**/*.tiff", recursive=True))
        np.random.shuffle(image_files)
        num_train_images = int(len(self.image_files)* split_percentage)

        if split == "train":
            self.image_files = self.image_files[:num_train_images]
        else:
            self.image_files = self.image_files[num_train_images:]

        self.classes = set()

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = rawpy.imread(image_path)
        raw_image = image.raw_image
        raw_image = raw_image.astype(np.float16)

        if self.transform:
            raw_image = self.transform(raw_image)
        else:
            image = ToTensor()(raw_image)

        label = Path(image_path).parent.stem
        return image, label

    def __len__(self) -> int:
        return len(self.image_files)