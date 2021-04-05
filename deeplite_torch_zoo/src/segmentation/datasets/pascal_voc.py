from functools import partial
from pathlib import Path

import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from deeplite_torch_zoo.src.objectdetection.configs import voc_config as cfg
from deeplite_torch_zoo.src.segmentation.datasets.utils.custum_aug import \
    PadIfNeededRightBottom
from deeplite_torch_zoo.src.segmentation.datasets.utils.preprocess import (
    meanstd_normalize, minmax_normalize)


class PascalVocDataset(Dataset):
    def __init__(
        self,
        base_dir="deeplite_torch_zoo/data/VOC/",
        split="train_aug",
        num_classes=21,
        affine_augmenter=None,
        image_augmenter=None,
        target_size=(512, 512),
        net_type="unet",
        ignore_index=255,
        debug=False,
    ):
        self.debug = debug
        #########To support subclasses######################
        self.classes = cfg.DATA["CLASSES"]
        self.all_classes = ["BACKGROUND"] + cfg.DATA["ALLCLASSES"]

        if num_classes == 2:
            self.classes = cfg.DATA["CLASSES_1"]
        elif num_classes == 3:
            self.classes = cfg.DATA["CLASSES_2"]

        self.classes = ["BACKGROUND"] + self.classes
        self.num_classes = len(self.classes)

        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}

        self.class_to_id_all = dict(zip(self.all_classes, range(len(self.all_classes))))

        self.map_selected_ids_to_all = {
            k: self.class_to_id_all[v] for k, v in self.id_to_class.items()
        }
        self.map_all_ids_to_selected = {
            v: k for k, v in self.map_selected_ids_to_all.items()
        }

        #########To support subclasses######################
        self.base_dir = Path(base_dir) / Path("VOCdevkit/VOC2012")
        self.net_type = net_type
        self.ignore_index = ignore_index
        self.split = split

        valid_ids = self.base_dir / "ImageSets" / "Segmentation" / "val.txt"
        with open(valid_ids, "r") as f:
            valid_ids = f.readlines()
        if self.split == "valid":
            lbl_dir = "SegmentationClass"
            img_ids = valid_ids
        else:
            valid_set = set([valid_id.strip() for valid_id in valid_ids])
            lbl_dir = "SegmentationClassAug" if "aug" in split else "SegmentationClass"
            all_set = set(
                [p.name[:-4] for p in self.base_dir.joinpath(lbl_dir).iterdir()]
            )
            img_ids = list(all_set - valid_set)

        img_ids = [_id for _id in img_ids if self._not_empty(_id, split)]

        self.img_paths = [
            (self.base_dir / "JPEGImages" / "{}.jpg".format(img_id.strip()))
            for img_id in img_ids
        ]
        self.lbl_paths = [
            (self.base_dir / lbl_dir / "{}.png".format(img_id.strip()))
            for img_id in img_ids
        ]

        # Resize
        if isinstance(target_size, str):
            target_size = eval(target_size)
        if "train" in self.split:
            if self.net_type == "deeplab":
                target_size = (target_size[0] + 1, target_size[1] + 1)
            self.resizer = albu.Compose(
                [
                    albu.RandomScale(scale_limit=(-0.5, 0.5), p=1.0),
                    PadIfNeededRightBottom(
                        min_height=target_size[0],
                        min_width=target_size[1],
                        value=0,
                        ignore_index=self.ignore_index,
                        p=1.0,
                    ),
                    albu.RandomCrop(height=target_size[0], width=target_size[1], p=1.0),
                ]
            )
        else:
            # self.resizer = None
            self.resizer = albu.Compose(
                [
                    PadIfNeededRightBottom(
                        min_height=target_size[0],
                        min_width=target_size[1],
                        value=0,
                        ignore_index=self.ignore_index,
                        p=1.0,
                    ),
                    albu.Crop(
                        x_min=0, x_max=target_size[1], y_min=0, y_max=target_size[0]
                    ),
                ]
            )

        # Augment
        if "train" in self.split:
            self.affine_augmenter = affine_augmenter
            self.image_augmenter = image_augmenter
        else:
            self.affine_augmenter = None
            self.image_augmenter = None

    def __len__(self):
        return len(self.img_paths)

    def filter_out_extra_labels(self, lbl):
        lbl = np.array(lbl, dtype=np.int32)
        for k_orig, k_selected in self.map_all_ids_to_selected.items():
            lbl[lbl == k_orig] = -k_selected
        lbl[lbl > 0] = 0
        return np.array(lbl * -1, dtype=np.uint8)

    def _not_empty(self, img_id, split):
        lbl_dir = "SegmentationClassAug" if "aug" in split else "SegmentationClass"
        lbl_path = self.base_dir / lbl_dir / "{}.png".format(img_id.strip())
        lbl = np.array(Image.open(lbl_path))
        lbl = self.filter_out_extra_labels(lbl)
        return len(np.unique(lbl)) > 1

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = np.array(Image.open(img_path))
        if self.split == "test":
            # Resize (Scale & Pad & Crop)
            if self.net_type == "unet":
                img = minmax_normalize(img)
                img = meanstd_normalize(
                    img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            else:
                img = minmax_normalize(img, norm_range=(-1, 1))
            if self.resizer:
                resized = self.resizer(image=img)
                img = resized["image"]
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img)
            return img
        else:
            lbl_path = self.lbl_paths[index]
            lbl = np.array(Image.open(lbl_path))
            # ImageAugment (RandomBrightness, AddNoise...)
            if self.image_augmenter:
                augmented = self.image_augmenter(image=img)
                img = augmented["image"]
            # Resize (Scale & Pad & Crop)
            if self.net_type == "unet":
                img = minmax_normalize(img)
                img = meanstd_normalize(
                    img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            else:
                img = minmax_normalize(img, norm_range=(-1, 1))
            if self.resizer:
                resized = self.resizer(image=img, mask=lbl)
                img, lbl = resized["image"], resized["mask"]
            # AffineAugment (Horizontal Flip, Rotate...)
            if self.affine_augmenter:
                augmented = self.affine_augmenter(image=img, mask=lbl)
                img, lbl = augmented["image"], augmented["mask"]
            lbl = self.filter_out_extra_labels(lbl)
            if self.debug:
                print(lbl_path)
                print(lbl.shape)
                print(np.unique(lbl))
            # else:
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img)
            lbl = torch.LongTensor(lbl)
            return img, lbl, img_path.stem


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.custum_aug import Rotate

    affine_augmenter = albu.Compose([albu.HorizontalFlip(p=0.5), Rotate(5, p=0.5)])
    # image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
    #                                 albu.RandomBrightnessContrast(p=.5)])
    image_augmenter = None
    dataset = PascalVocDataset(
        affine_augmenter=affine_augmenter,
        image_augmenter=image_augmenter,
        split="valid",
        net_type="deeplab",
        ignore_index=21,
        target_size=(512, 512),
        debug=True,
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataset))

    for i, batched in enumerate(dataloader):
        images, labels, _ = batched
        if i == 0:
            fig, axes = plt.subplots(8, 2, figsize=(20, 48))
            plt.tight_layout()
            for j in range(8):
                axes[j][0].imshow(
                    minmax_normalize(images[j], norm_range=(0, 1), orig_range=(-1, 1))
                )
                axes[j][1].imshow(labels[j])
                axes[j][0].set_xticks([])
                axes[j][0].set_yticks([])
                axes[j][1].set_xticks([])
                axes[j][1].set_yticks([])
            plt.savefig("dataset/pascal_voc.png")
            plt.close()
        break
