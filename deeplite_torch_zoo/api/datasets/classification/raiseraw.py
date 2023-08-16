import os
from os.path import expanduser
import glob
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import rawpy

from deeplite_torch_zoo.api.datasets.utils import get_dataloader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY


__all__ = ["get_raise_6k"]


@DATASET_WRAPPER_REGISTRY.register(dataset_name="raw_raise6k")
def get_raise_6k(
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
                 transform = None,
                 split_percentage=0.8,
                 ):
        
        self.data_dir = root
        self.transform = transform
        self.split = split

        self.metadata = pd.read_csv(metadata_file)
        self.image_files = glob.glob(os.path.join(root, "*.NEF"))
        num_train_images = int(len(self.image_files)* split_percentage)
        if split == "train":
            self.image_files = self.image_files[:num_train_images]
        else:
            self.image_files = self.image_files[num_train_images:]

        self.classes = set()
        
    def __len__(self):
        return len(self.image_files)

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
