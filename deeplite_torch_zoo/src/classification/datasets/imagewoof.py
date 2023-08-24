import os
from pathlib import Path

import PIL.Image
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


IMAGEWOOF_IMAGENET_CLS_LABEL_MAP = (155, 159, 162, 167, 182, 193, 207, 229, 258, 273)


class Imagewoof(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        download: bool = False,
        url: str = "",
        map_to_imagenet_labels=False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val"))
        self.url = url
        self._base_folder = Path(self.root) / url.split('/')[-1].replace('.zip', '')

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self._labels = []
        self._image_files = []

        self.classes = sorted(
            entry.name
            for entry in os.scandir(self._base_folder / "train")
            if entry.is_dir()
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self._base_folder / f"{split}", target_class)
            for folder, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(folder, fname)
                    self._labels.append(class_index)
                    self._image_files.append(path)

        self._map_to_imagenet_labels = map_to_imagenet_labels

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self._map_to_imagenet_labels:
            label = IMAGEWOOF_IMAGENET_CLS_LABEL_MAP[label]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(
            folder.exists() and folder.is_dir() for folder in (self._base_folder,)
        )

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self.url, download_root=self.root)
