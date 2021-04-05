import logging
import os
import pathlib
import xml.etree.ElementTree as ET

import cv2
import numpy as np

import deeplite_torch_zoo.src.objectdetection.configs.voc_config as cfg

VOC_CLASS_NAMES = {
    "BACKGROUND",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
}


class VOCDataset:
    def __init__(
        self,
        root,
        n_classes=21,
        transform=None,
        target_transform=None,
        is_test=False,
        keep_difficult=False,
        label_file=None,
    ):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.keep_difficult = keep_difficult

        self.classes = cfg.DATA["CLASSES"]
        self.all_classes = ["BACKGROUND"] + cfg.DATA["ALLCLASSES"]

        if n_classes == 1:
            self.classes = cfg.DATA["CLASSES_1"]
        elif n_classes == 2:
            self.classes = cfg.DATA["CLASSES_2"]

        self.classes = ["BACKGROUND"] + self.classes
        self.num_classes = len(self.classes)

        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}

        self.class_to_id_all = dict(zip(self.all_classes, range(len(self.all_classes))))

        self.map_selected_ids_to_all = {
            k: self.class_to_id_all[v] for k, v in self.id_to_class.items()
        }

        self.ids = self._read_image_ids(image_sets_file, is_test)

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, _, _ = self.transform(image, boxes, labels)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _read_image_ids(self, image_sets_file, is_test=False):
        ids = []
        with open(str(image_sets_file)) as f:
            for line in f:
                _id = line.rstrip()
                bboxes = self.__not_empty(_id)
                if is_test or (len(bboxes) > 0 and bboxes.shape[0] > 0):
                    ids.append(_id)
        return ids

    def __not_empty(self, image_id):
        bboxes, _, is_difficult = self._get_annotation(image_id)
        bboxes = bboxes[is_difficult == 0]
        return bboxes

    def _get_annotation(self, image_id):
        annotation_file = self.root / "Annotations/{}.xml".format(image_id)
        objects = ET.parse(str(annotation_file)).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find("name").text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_to_id:
                bbox = object.find("bndbox")

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find("xmin").text) - 1
                y1 = float(bbox.find("ymin").text) - 1
                x2 = float(bbox.find("xmax").text) - 1
                y2 = float(bbox.find("ymax").text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_to_id[class_name])
                is_difficult_str = object.find("difficult").text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(is_difficult, dtype=np.uint8),
        )

    def _read_image(self, image_id):
        image_file = self.root / "JPEGImages/{}.jpg".format(image_id)
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image