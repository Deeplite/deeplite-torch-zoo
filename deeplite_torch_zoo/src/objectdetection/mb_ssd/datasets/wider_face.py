import os
import pathlib
import cv2
import numpy as np
import torch
from os.path import abspath, expanduser
from typing import Any, Callable, List, Dict, Optional, Tuple, Union


WF_CLASS_NAMES = {
    "BACKGROUND",
    "face",
}


class WiderFace:
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        split="train",
    ):
        """`WIDERFace <http://shuoyang1213.me/WIDERFACE/>`_ Dataset.
        Args:
            root (string): Root directory where images and annotations are downloaded to.
                Expects the following folder structure:
                .. code::
                    <root>
                        └── widerface
                            ├── wider_face_split 
                            ├── WIDER_train
                            ├── WIDER_val 
                            └── WIDER_test
            split (string): The dataset split to use. One of {``train``, ``val``, ``test``}.
                Defaults to ``train``.
            transform (callable, optional): A function/transform that  takes in an image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it in ssd targets format.
        """
        self.root = pathlib.Path(root)
        self.transform = transform

        self.split = split
        self.img_info: List[Dict[str, Union[str, Dict[str, torch.Tensor]]]] = []

        if self.split == "test":
            self.parse_test_annotations_file()
        else:
            self.parse_train_val_annotations_file()

        print(f"{split}: {len(self.img_info)}")
        self.target_transform = target_transform
        self.classes = WF_CLASS_NAMES
        self.num_classes = len(self.classes)

        self.ids = [str(i) for i in range(0, len(self.img_info))]

    def __getitem__(self, index):
        boxes, labels, is_difficult = self._get_annotation(index)
        image = self._read_image(index)
        boxes = boxes[is_difficult == 0]
        labels = labels[is_difficult == 0]
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        if torch.is_tensor(boxes):
            return image, boxes.type(torch.float32), labels.type(torch.LongTensor)
        return image, boxes, labels

    def get_image(self, index):
        boxes, labels, _ = self._get_annotation(index)
        image = self._read_image(index)
        if self.transform:
            image, _, _ = self.transform(image, boxes, labels)
        return image

    def get_annotation(self, index):
        return str(index), self._get_annotation(index)

    def _get_annotation(self, index):
        boxes = self.img_info[index]["annotations"]["bbox"]
        labels = np.ones(len(boxes))
        return boxes, labels, self.img_info[index]["annotations"]["invalid"]

    def _read_image(self, index):
        image_file = self.img_info[index]["img_path"]
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self) -> int:
        return len(self.img_info)

    def _parse_file_name(self, line):
        line = line.rstrip()
        img_path = os.path.join(self.root, "WIDER_" + self.split, "images", line)
        img_path = abspath(expanduser(img_path))
        return img_path

    def parse_train_val_annotations_file(self) -> None:
        filename = "wider_face_train_bbx_gt.txt" if self.split == "train" else "wider_face_val_bbx_gt.txt"
        filepath = os.path.join(self.root, "wider_face_split", filename)

        with open(filepath, "r") as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                labels = []
                img_path = self._parse_file_name(lines[i])
                _num_boxes = int(lines[i+1].rstrip())
                num_boxes = max(1, _num_boxes)
                i = i + 2
                bboxes_lines = lines[i: i+num_boxes]
                i = i + num_boxes
                for bbox_line in bboxes_lines:
                    line_split = bbox_line.rstrip().split(" ")
                    bbox_values = np.array([float(x) for x in line_split])

                    # Tiny faces cause inf loss for bboxes regression loss
                    if bbox_values[2] < 20 or bbox_values[3] < 20:
                        continue
                    bbox_values[2:4] += bbox_values[0:2]
                    labels.append(bbox_values)

                if len(labels) == 0:
                    continue

                labels = np.array(labels)
                _invalid = labels[:, 7]
                valid_boxes = num_boxes - sum(_invalid)
                if _num_boxes == 0 or valid_boxes <= 0:
                    continue

                self.img_info.append({
                    "img_path": img_path,
                    "annotations": {"bbox": labels[:, 0:4],  # xmin, ymin, xmax, ymax
                                    "blur": labels[:, 4],
                                    "expression": labels[:, 5],
                                    "illumination": labels[:, 6],
                                    "invalid": _invalid,
                                    "occlusion": labels[:, 8],
                                    "pose": labels[:, 9]}
                })

    def parse_test_annotations_file(self) -> None:
        filepath = os.path.join(self.root, "wider_face_split", "wider_face_test_filelist.txt")
        filepath = abspath(expanduser(filepath))
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                img_path = os.path.join(self.root, "WIDER_test", "images", line)
                img_path = abspath(expanduser(img_path))
                self.img_info.append({"img_path": img_path})
