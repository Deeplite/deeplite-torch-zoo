import os
from os.path import expanduser
from pathlib import Path

import PIL.Image
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
    verify_str_arg,
)
from torchvision.datasets.vision import VisionDataset

from deeplite_torch_zoo.src.classification.augmentations.augs import (
    get_imagenet_transforms,
    get_vanilla_transforms,
)
from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY

__all__ = ["get_flowers102"]


@DATASET_WRAPPER_REGISTRY.register(dataset_name="flowers102")
def get_flowers102(
    data_root="",
    batch_size=64,
    test_batch_size=None,
    img_size=224,
    num_workers=4,
    fp16=False,
    download=True,
    distributed=False,
    augmentation_mode='imagenet',
    train_transforms=None,
    val_transforms=None,
    **kwargs,
):
    if data_root == "":
        data_root = os.path.join(expanduser("~"), ".deeplite-torch-zoo")

    if augmentation_mode not in ('vanilla', 'imagenet'):
        raise ValueError(
            f'Wrong value of augmentation_mode arg: {augmentation_mode}. Choices: "vanilla", "imagenet"'
        )

    if augmentation_mode == 'imagenet':
        default_train_transforms, default_val_transforms = get_imagenet_transforms(
            img_size
        )
    else:
        default_train_transforms, default_val_transforms = get_vanilla_transforms(
            img_size
        )

    train_transforms = (
        train_transforms if train_transforms is not None else default_train_transforms
    )
    val_transforms = (
        val_transforms if val_transforms is not None else default_val_transforms
    )

    train_dataset = Flowers102(
        root=data_root, split='train', download=download, transform=train_transforms
    )

    test_dataset = Flowers102(
        root=data_root,
        split='test',
        download=download,
        transform=val_transforms,
    )

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


class Flowers102(VisionDataset):
    # Taken from https://github.com/pytorch/vision/blob/HEAD/torchvision/datasets/flowers102.py
    # Added for compatibility with old torchvision versions

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        from scipy.io import loadmat

        set_ids = loadmat(
            self._base_folder / self._file_dict["setid"][0], squeeze_me=True
        )
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(
            self._base_folder / self._file_dict["label"][0], squeeze_me=True
        )
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

        self.classes = set(self._labels)

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

    def extra_repr(self):
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(
                self._download_url_prefix + filename, str(self._base_folder), md5=md5
            )
