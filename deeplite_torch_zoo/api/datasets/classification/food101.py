import json
import os
from os.path import expanduser
from pathlib import Path

import PIL.Image
from torchvision.datasets.utils import (download_and_extract_archive,
                                        verify_str_arg)
from torchvision.datasets.vision import VisionDataset

from deeplite_torch_zoo.src.classification.augmentations.augs import (
    get_imagenet_transforms, get_vanilla_transforms)
from deeplite_torch_zoo.api.datasets.utils import get_dataloader
from deeplite_torch_zoo.api.registries import DATA_WRAPPER_REGISTRY

__all__ = ["get_food101"]


@DATA_WRAPPER_REGISTRY.register(dataset_name="food101")
def get_food101(
    data_root="", batch_size=64, test_batch_size=None, img_size=224, num_workers=4,
    fp16=False, download=True, device="cuda", distributed=False,
    augmentation_mode='imagenet', train_transforms=None, val_transforms=None, **kwargs,
):
    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")

    if augmentation_mode not in ('vanilla', 'imagenet'):
        raise ValueError(f'Wrong value of augmentation_mode arg: {augmentation_mode}. Choices: "vanilla", "imagenet"')

    if augmentation_mode == 'imagenet':
        default_train_transforms, default_val_transforms = get_imagenet_transforms(img_size)
    else:
        default_train_transforms, default_val_transforms = get_vanilla_transforms(img_size)

    train_transforms = train_transforms if train_transforms is not None else default_train_transforms
    val_transforms = val_transforms if val_transforms is not None else default_val_transforms

    train_dataset = Food101(
        root=data_root,
        split='train',
        download=download,
        transform=train_transforms
    )

    test_dataset = Food101(
        root=data_root,
        split='test',
        download=download,
        transform=val_transforms,
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_batch_size = batch_size if test_batch_size is None else test_batch_size
    test_loader = get_dataloader(test_dataset, batch_size=test_batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "test": test_loader}


class Food101(VisionDataset):
    # Taken from https://github.com/pytorch/vision/blob/HEAD/torchvision/datasets/food101.py
    # Added for compatibility with old torchvision versions

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform = None,
        target_transform = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json", encoding="utf8") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)
