import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import deeplite_torch_zoo.src.objectdetection.configs.lisa_config as lisa_cfg
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.data_augment import (
    Mixup, RandomAffine, RandomCrop, RandomHorizontalFilp, Resize)


class LISA(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self, data_folder="data/lisa", _set="train", split=0.7, seed=123, img_size=416
    ):
        """
        :param data_folder: folder where data files are stored
        :param split: split data randomly with split % for training and 1 - split % for validation
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """

        self.data_folder = Path(data_folder)
        self.images = []
        self.objects = []
        self.data = {}
        self.classes = lisa_cfg.DATA["CLASSES"]
        self._img_size = img_size
        self.label_map = {}
        self.inv_map = {}
        self._set = _set

        self.split = split
        random.seed(seed)
        headers = 1
        with open(self.data_folder / "allAnnotations.csv") as annotations:
            csv_reader = csv.reader(annotations, delimiter=";")
            for _sample in csv_reader:
                img_path = Path(_sample[0])
                if headers:
                    headers = 0
                    continue
                label = _sample[1]

                if label not in self.classes:
                    # print(f'skipping {label}')
                    continue
                bbox = [_sample[2], _sample[3], _sample[4], _sample[5]]
                bbox = [int(c) for c in bbox]

                if img_path not in self.data:
                    self.data[img_path] = {"boxes": [], "labels": []}

                self.data[img_path]["boxes"].append(bbox)
                self.data[img_path]["labels"].append(label)

        self.images = list(self.data.keys())
        self.objects = list(self.data.values())
        self.label_map = {k: v for v, k in enumerate(sorted(list(self.classes)))}
        self.inv_map = {k: v for k, v in enumerate(sorted(list(self.classes)))}
        num_images = len(self.images)

        shuffled_indices = list(range(num_images))
        random.shuffle(shuffled_indices)
        self._set = _set.lower()
        assert self._set in self.available_sets

        num_training_samples = int(self.split * num_images)

        if self._set == "train":
            train_indices = shuffled_indices[:num_training_samples]
            self.images = self._subsample(self.images, train_indices)
            self.objects = self._subsample(self.objects, train_indices)

        else:
            valid_indices = shuffled_indices[num_training_samples:]
            self.images = self._subsample(self.images, valid_indices)
            self.objects = self._subsample(self.objects, valid_indices)

        assert len(self.images) == len(self.objects)

    @property
    def available_sets(self):
        return ["train", "valid"]

    @property
    def num_classes(self):
        return len(self.classes)

    def _subsample(self, samples, indices):
        return [samples[idx] for idx in indices]

    def _augment(self, img, bboxes):
        img, bboxes = RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = Resize((self._img_size, self._img_size), True)(
            np.copy(img), np.copy(bboxes)
        )
        return img, bboxes

    def _get_image(self, idx):
        file_path = str(self.data_folder / self.images[idx])
        img = cv2.imread(file_path)
        objects = self.objects[idx]
        labels = [self.label_map[label] for label in objects["labels"]]
        boxes = objects["boxes"]
        bboxes = np.zeros((len(boxes), 5))
        bboxes[:, :4] = boxes
        img, bboxes = self._augment(img, bboxes)
        bboxes[:, 4] = labels
        return img, bboxes

    def __getitem__(self, i):
        # Read image
        img_org, bboxes_org = self._get_image(i)
        if self._set == "valid":
            return img_org, bboxes_org, bboxes_org.shape[0], 0
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.images) - 1)

        img_mix, bboxes_mix = self._get_image(item_mix)
        img_mix = img_mix.transpose(2, 0, 1)  # HWC->CHW
        img, bboxes = Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix

        img = torch.from_numpy(img).float()

        bboxes = torch.from_numpy(bboxes).float()

        return img, bboxes, bboxes.shape[0], 0

    def __len__(self):
        """
        :return: number of images in the dataset.
        """
        return len(self.images)

    def collate_fn(self, sample):
        images = []
        labels = []
        lengths = []
        labels_with_tail = []
        img_ids = []
        max_num_obj = 0
        for image, label, length, img_id in sample:
            images.append(image)
            labels.append(label)
            lengths.append(length)
            img_ids.append(torch.tensor([img_id]))
            max_num_obj = max(max_num_obj, length)
        for label in labels:
            num_obj = label.size(0)
            zero_tail = torch.zeros(
                (max_num_obj - num_obj, label.size(1)),
                dtype=label.dtype,
                device=label.device,
            )
            label_with_tail = torch.cat((label, zero_tail), dim=0)
            labels_with_tail.append(label_with_tail)
        image_tensor = torch.stack(images)
        label_tensor = torch.stack(labels_with_tail)
        length_tensor = torch.tensor(lengths)
        img_ids_tensor = torch.stack(img_ids)
        return image_tensor, label_tensor, length_tensor, img_ids_tensor

