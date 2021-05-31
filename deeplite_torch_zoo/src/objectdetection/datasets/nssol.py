import cv2
import os
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from deeplite_torch_zoo.src.objectdetection.eval.coco.utils import xywh_to_xyxy
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.data_augment import (
    Mixup, RandomAffine, RandomCrop, RandomHorizontalFilp, Resize)


class NSSOLDataset(Dataset):
    def __init__(self, data_root, _set="train", img_size=448):

        classes_file = os.path.join(data_root, "classes.txt")
        setfile = os.path.join(data_root, f"{_set}.txt")
        self._img_size = img_size
        self.data_root = data_root
        self._set = _set
        self.classes = self.__load_classes(classes_file)
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.setfile = setfile
        self.__elements = self.__load_available_data(setfile)

        print(f"Dataset created. Resize to {self._img_size}. Using {self.num_classes} classes. Dataset {self.setfile}")

    def __load_classes(self, classes_file):
        assert Path(classes_file).exists(), f"{classes_file} does not exist!"
        lines = Path(classes_file).open("r").readlines()

        return [line.strip() for line in lines if len(line) > 0]

    def __len__(self):
        return len(self.__elements)

    def __getitem__(self, i):
        # Read image
        img_org, bboxes_org = self._get_image(i)
        img_id = self.__elements[i][1]
        if self._set is not "train":
            return img_org, bboxes_org, bboxes_org.shape[0], img_id

        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW
        item_mix = random.randint(0, len(self.__elements) - 1)
        img_mix, bboxes_mix = self._get_image(item_mix)
        img_mix = img_mix.transpose(2, 0, 1)  # HWC->CHW
        img, bboxes = Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix

        img = torch.from_numpy(img).float()
        bboxes = torch.from_numpy(bboxes).float()
        return img, bboxes, bboxes.shape[0], self.__elements[i][1]

    def collate_img_label_fn(self, sample):
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
            zero_tail = torch.zeros((max_num_obj - num_obj, label.size(1)), dtype=label.dtype, device=label.device)
            label_with_tail = torch.cat((label, zero_tail), dim=0)
            labels_with_tail.append(label_with_tail)
        image_tensor = torch.stack(images)
        label_tensor = torch.stack(labels_with_tail)
        length_tensor = torch.tensor(lengths)
        img_ids_tensor = torch.stack(img_ids)
        return image_tensor, label_tensor, length_tensor, img_ids_tensor

    def __load_available_data(self, setfile):
        assert Path(setfile).exists(), f"{setfile} not found!"
        lines = Path(setfile).open('r').readlines()
        result = [(line.strip(), id) for id, line in enumerate(lines) if len(line) > 0]
        assert len(result) > 0, "No input data found!"
        return result

    def _augment(self, img, bboxes):
        img, bboxes = RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = Resize((self._img_size, self._img_size), True)(
            np.copy(img), np.copy(bboxes)
        )
        return img, bboxes

    def _get_image(self, idx):
        file_path = self.__elements[idx][0]
        file_path = os.path.join(self.data_root, 'images/', os.path.basename(file_path))
        img = cv2.imread(file_path)
        h, w, _ = img.shape
        assert img is not None, f'File Not Found {file_path}'
        anno_file = Path(file_path).with_suffix(".txt")

        assert anno_file.exists(), f'File Not Found {anno_file}'
        annotations = anno_file.open('r').readlines()
        anno = np.array([x.split() for x in annotations], dtype=np.float32)

        bboxes = np.zeros((len(anno), 5))
        bboxes[:, :4] = anno[:, 1:]
        bboxes[:, 0] = bboxes[:, 0] * w
        bboxes[:, 2] = bboxes[:, 2] * w
        bboxes[:, 1] = bboxes[:, 1] * h
        bboxes[:, 3] = bboxes[:, 3] * h
        bboxes = xywh_to_xyxy(bboxes)
        bboxes[bboxes[:, 2] >= w, 2] = w - 1
        bboxes[bboxes[:, 3] >= h, 3] = h -1

        if self._set == "train":
            img, bboxes = self._augment(img, bboxes)
        bboxes[:, 4] = anno[:, 0]
        return img, bboxes

