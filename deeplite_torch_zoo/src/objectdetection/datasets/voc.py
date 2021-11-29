import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import deeplite_torch_zoo.src.objectdetection.configs.voc_config as cfg
from deeplite_torch_zoo.src.objectdetection.datasets.data_augment import (
    Mixup, RandomAffine, RandomCrop, RandomHorizontalFlip, Resize,
    AugmentHSV, Albumentations)


class VocDataset(Dataset):
    def __init__(self, annotation_path, anno_file_type, num_classes=None,
                    img_size=416, mosaic=False, version_6_augs=False):
        self._img_size = img_size  # for multi-scale training
        self.classes = cfg.DATA["CLASSES"]
        if num_classes == 1:
            self.classes = cfg.DATA["CLASSES_1"]
        elif num_classes == 2:
            self.classes = cfg.DATA["CLASSES_2"]

        self.all_classes = cfg.DATA["ALLCLASSES"]
        self.annotation_path = annotation_path
        self.num_classes = len(self.classes)

        self.mosaic = mosaic
        self.version_6_augs = version_6_augs

        if num_classes is not None:
            self.num_classes = num_classes

        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}

        self.class_to_id_all = dict(zip(self.all_classes, range(len(self.all_classes))))

        self.map_selected_ids_to_all = {
            k: self.class_to_id_all[v] for k, v in self.id_to_class.items()
        }
        self.map_all_ids_to_selected = {
            v: k for k, v in self.map_selected_ids_to_all.items()
        }

        self.__annotations = self.__load_annotations(anno_file_type)
        if anno_file_type == "train":
            self.__annotations = [
                annotation
                for annotation in self.__annotations
                if self.__not_empty(annotation)
            ]

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):
        """
        return:
            bboxes of shape nx6, where n is number of labels in the image and x1,y1,x2,y2, class_id and confidence.
        """

        if self.mosaic:
            img, bboxes, img_id = self.load_mosaic(item)
        else:
            # MixUp
            img_org, bboxes_org, img_id = self.__parse_annotation(self.__annotations[item])
            img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

            item_mix = random.randint(0, len(self.__annotations) - 1)
            img_mix, bboxes_mix, _ = self.__parse_annotation(self.__annotations[item_mix])
            img_mix = img_mix.transpose(2, 0, 1)

            img, bboxes = Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
            del img_org, bboxes_org, img_mix, bboxes_mix

        img = torch.from_numpy(img).float()
        bboxes = torch.from_numpy(bboxes).float()

        return img, bboxes, bboxes.shape[0], img_id

    def collate_img_label_fn(self, sample):
        images = []
        labels = []
        lengths = []
        labels_with_tail = []
        max_num_obj = 0
        for image, label, length, img_id in sample:
            images.append(image)
            labels.append(label)
            lengths.append(length)
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
        return image_tensor, label_tensor, length_tensor, None

    def __load_annotations(self, anno_type):

        assert anno_type in [
            "train",
            "test",
        ], "You must choice one of the 'train' or 'test' for anno_type parameter"
        anno_path = os.path.join(self.annotation_path, anno_type + "_annotation.txt")
        with open(anno_path, "r") as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        assert len(annotations) > 0, "No images found in {}".format(anno_path)

        return annotations

    def filter_out_extra_classes(self, bboxes_):
        bboxes = []
        for bbox in bboxes_:
            if bbox[4] in self.map_all_ids_to_selected:
                bbox[4] = self.map_all_ids_to_selected[bbox[4]]
                bboxes.append(list(bbox))
        return np.array(bboxes)

    def __not_empty(self, annotation):
        anno = annotation.strip().split(" ")
        bboxes = np.array([list(map(float, box.split(","))) for box in anno[1:]])
        bboxes = self.filter_out_extra_classes(bboxes)
        return len(bboxes)

    def __parse_annotation(self, annotation):
        """
        Data augument.
        :param annotation: Image' path and bboxes' coordinates, categories.
        ex. [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]
        :return: Return the enhanced image and bboxes. bbox'shape is [xmin, ymin, xmax, ymax, class_ind]
        """
        anno = annotation.strip().split(" ")

        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, "File Not Found " + img_path
        bboxes = np.array([list(map(float, box.split(","))) for box in anno[1:]])
        bboxes = self.filter_out_extra_classes(bboxes)

        if len(bboxes) == 0:
            bboxes = np.array(np.zeros((0, 5)))
        else:
            img, bboxes = RandomHorizontalFlip()(np.copy(img), np.copy(bboxes))
            if self.version_6_augs:
                img, bboxes = Albumentations()(np.copy(img), np.copy(bboxes))
                img = AugmentHSV()(np.copy(img))
            img, bboxes = RandomCrop()(np.copy(img), np.copy(bboxes))
            img, bboxes = RandomAffine()(np.copy(img), np.copy(bboxes))

        img, bboxes = Resize((self._img_size, self._img_size), True)(
            np.copy(img), np.copy(bboxes)
        )
        return img, bboxes, str(Path(img_path).stem)

    def load_mosaic(self, item, resize_to_original_size=False):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        bboxes4 = []
        s = self._img_size
        mosaic_border = [-s // 2, -s // 2]
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y
        indices = [item] + random.choices(range(0, len(self.__annotations)), k=3)  # 3 additional image indices
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load image
            img, bboxes, img_id = self.__parse_annotation(self.__annotations[index])
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114.0, dtype=np.float32) #114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            if bboxes.size:
                bboxes[:, 0] += padw
                bboxes[:, 2] += padw
                bboxes[:, 1] += padh
                bboxes[:, 3] += padh

            bboxes4.append(bboxes)

        # Concat/clip labels
        bboxes4 = np.concatenate(bboxes4, 0)
        bboxes4 = np.clip(bboxes4, 0, img4.shape[0])

        if resize_to_original_size:
            img4, bboxes4 = Resize((s, s), True, False)(
                np.copy(img4), np.copy(bboxes4)
            )

        img4 = img4.transpose(2, 0, 1)  # HWC->CHW
        return img4, bboxes4, img_id


if __name__ == "__main__":

    voc_dataset = VocDataset(anno_file_type="train", img_size=448)
    dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0)
