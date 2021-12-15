import os
from pathlib import Path

import random
import cv2
import numpy as np
import torch

from deeplite_torch_zoo.src.objectdetection.datasets.dataset import DLZooDataset
import deeplite_torch_zoo.src.objectdetection.configs.hyps.hyp_config_voc as cfg
from deeplite_torch_zoo.src.objectdetection.datasets.data_augment import Resize


class VocDataset(DLZooDataset):
    def __init__(self, annotation_path, anno_file_type, num_classes=None, img_size=416):
        super().__init__(cfg.TRAIN, img_size)
        self.classes = cfg.DATA["CLASSES"]
        if num_classes == 1:
            self.classes = cfg.DATA["CLASSES_1"]
        elif num_classes == 2:
            self.classes = cfg.DATA["CLASSES_2"]

        self.all_classes = cfg.DATA["ALLCLASSES"]
        self.annotation_path = annotation_path
        self.num_classes = len(self.classes)

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

        get_img_fn = lambda img_index: self.__parse_annotation(self.__annotations[img_index])
        if random.random() < cfg.TRAIN['mosaic']:
            img, bboxes, img_id = self._load_mosaic(item, get_img_fn,
                len(self.__annotations))
        img, bboxes, img_id = self._load_mixup(item, get_img_fn,
            len(self.__annotations), p=cfg.TRAIN['mixup'])

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
            img, bboxes = self._augment(img, bboxes)

        img, bboxes = Resize((self._img_size, self._img_size), True)(
            np.copy(img), np.copy(bboxes)
        )
        return img, bboxes, str(Path(img_path).stem)
