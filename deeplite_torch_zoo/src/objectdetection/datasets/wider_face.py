import os
from pathlib import Path

import cv2
import numpy as np
import torch
from os.path import abspath, expanduser
from typing import List, Dict, Union

from deeplite_torch_zoo.src.objectdetection.datasets.data_augment import Resize
from deeplite_torch_zoo.src.objectdetection.datasets.dataset import DLZooDataset
import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_default as cfg


WF_CLASS_NAMES = {
    "BACKGROUND",
    "face",
}


class WiderFace(DLZooDataset):
    def __init__(self, root, split="train", num_classes=None, img_size=416):
        super().__init__(cfg.TRAIN, img_size)
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

        """
        self.root = Path(root)
        self.split = split
        self.img_info: List[Dict[str, Union[str, Dict[str, torch.Tensor]]]] = []

        if self.split == "test":
            self.parse_test_annotations_file()
        else:
            self.parse_train_val_annotations_file()

        print(f"{split}: {len(self.img_info)}")
        self.classes = WF_CLASS_NAMES
        self.num_classes = len(self.classes)
        if num_classes is not None:
            self.num_classes = num_classes
        self.inv_map = {k: v for k, v in enumerate(sorted(list(self.classes)))}

    def __getitem__(self, item):
        """
        return:
            bboxes of shape nx6, where n is number of labels in the image and x1,y1,x2,y2, class_id and confidence.
        """

        get_img_fn = lambda img_index: self.__parse_annotation(self.img_info[img_index])
        img, bboxes, img_id = self._load_mixup(item, get_img_fn,
            len(self.img_info), p=cfg.TRAIN['mixup'])

        img = torch.from_numpy(img).float()
        bboxes = torch.from_numpy(bboxes).float()

        return img, bboxes, bboxes.shape[0], img_id

    def __parse_annotation(self, _info):
        """
        Data augument.
        :param annotation: Image' path and bboxes' coordinates, categories.
        ex. [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]
        :return: Return the enhanced image and bboxes. bbox'shape is [xmin, ymin, xmax, ymax, class_ind]
        """

        img_path = _info["img_path"]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, "File Not Found " + img_path

        bboxes = _info["annotations"]["bbox"]
        labels = np.ones((len(bboxes), 1))
        bboxes = np.concatenate((bboxes, labels), axis=1)

        if len(bboxes) == 0:
            bboxes = np.array(np.zeros((0, 5)))
        else:
            img, bboxes = self._augment(img, bboxes)
        img, bboxes = Resize((self._img_size, self._img_size), True)(
            np.copy(img), np.copy(bboxes)
        )
        return img, bboxes, str(Path(img_path).stem)

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
