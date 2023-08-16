import json
import os
from os.path import expanduser
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import rawpy

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from deeplite_torch_zoo.api.datasets.utils import get_dataloader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY


__all__ = ["get_raise_6k"]


@DATASET_WRAPPER_REGISTRY.register(dataset_name="raw_raise6k")
def get_raise_6k(
    data_root="",
    batch_size=64,
    test_batch_size=None,
    img_size=224,
    num_workers=4,
    fp16=False,
    download=True,
    distributed=False,
    # augmentation_mode='imagenet',
    train_transforms=None,
    val_transforms=None,
    **kwargs,
):
    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")
    
    metadata_file = os.path.join(data_root, "RAISE_6k.csv")
    image_folder = os.path.join(data_root, "NEF")
    train_dataset = RAISEDataset(
        root=image_folder, metadata_file=metadata_file,
        split='train', transform=train_transforms)

    test_dataset = RAISEDataset(
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


class RAISEDataset(Dataset):
    def __init__(self, 
                 root, 
                 metadata_file, 
                 split: str = "train",
                 transform = None
                 ):
        
        self.data_dir = root
        self.transform = transform
        self.split = split

        self.metadata = pd.read_csv(metadata_file)
        self.image_files = glob.glob(os.path.join(root, "*.NEF"))
        if split == "train":
            self.image_files = self.image_files[:4000]
        else:
            self.image_files = self.image_files[4000:]

        self.classes = set()
        
    def __len__(self):
        return len(self.image_files)

    # def get_rgb(self, arr):
    #     r = np.zeros_like(arr)
    #     g = np.zeros_like(arr)
    #     b = np.zeros_like(arr)
    #     r[::2,::2]  += arr[::2,::2]
    #     g[::2,1::2] += arr[::2,1::2]
    #     g[1::2,::2] += arr[1::2,::2]
    #     b[1::2,1::2]  += arr[1::2,1::2]
    #     rgb = cv2.merge((r, g, b))
    #     return rgb

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        image = rawpy.imread(image_path)
        raw_image = image.raw_image
        raw_image = raw_image.astype(np.float16)
        
        if self.transform:
            raw_image = self.transform(raw_image)
        else:
            image = ToTensor()(raw_image)

        keywords = (self.metadata[self.metadata['File'] == image_id]["Keywords"]).squeeze()
        labels_name = keywords.split(";")
        label = 0 if "outdoor" in labels_name else 1
        return image, label
    
    def __len__(self) -> int:
        return len(self.image_files)


if __name__ == "__main__":
    # Example usage:
    data_dir = "/home/shahrukh/project/raw_process/raw_images/RAISE_2k_NEF"
    metadata_file = "/home/shahrukh/project/raw_process/raw_images/RAISE_2k.csv"
    dataset = RAISEDataset(data_dir, metadata_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)