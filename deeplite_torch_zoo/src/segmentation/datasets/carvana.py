import logging
from glob import glob
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix="_mask", img_size=512):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.num_classes = 2
        self.img_size = img_size
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.ids = [
            splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith(".")
        ]
        logging.info("Creating dataset with {} examples".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img, is_img=True):
        w, h = pil_img.size
        newW, newH = self.img_size, self.img_size
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1 and is_img:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + ".*")
        img_file = glob(self.imgs_dir + idx + ".*")

        assert (
            len(mask_file) == 1
        ), "Either no mask or multiple masks found for the ID {}: {}".format(
            idx, mask_file
        )
        assert (
            len(img_file) == 1
        ), "Either no image or multiple images found for the ID {}: {}".format(
            idx, img_file
        )
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert (
            img.size == mask.size
        ), "Image and mask {idx} should be the same size, but are {} and {}".format(
            img.size, mask.size
        )

        img = self.preprocess(img)
        mask = self.preprocess(mask, is_img=False)
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(mask).float(),
            Path(img_file[0]).stem,
        )
        # return {
        #    'image': torch.from_numpy(img).type(torch.FloatTensor),
        #    'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        # }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix="_mask")
