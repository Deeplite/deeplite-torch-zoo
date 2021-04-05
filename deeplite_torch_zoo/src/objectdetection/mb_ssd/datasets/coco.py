# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import numpy as np
import torch
import cv2

from PIL import ImageFile
from torchvision.datasets import CocoDetection

from deeplite_torch_zoo.src.objectdetection.eval.coco.utils import xywh_to_xyxy


class CocoDetectionBoundingBox(CocoDetection):
    def __init__(
        self,
        img_root,
        ann_file_name,
        transform=None,
        target_transform=None,
        category="all",
        missing_ids=[],
        classes=[],
    ):
        super(CocoDetectionBoundingBox, self).__init__(img_root, ann_file_name)
        self.missing_ids = missing_ids
        self.transform = transform
        self.target_transform = target_transform

        self.classes = ["BACKGROUND"] + classes
        self.class_names = self.classes
        self.num_classes = len(self.classes)
        if category == "all":
            self.all_categories = True
            self.category_id = -1
        elif isinstance(category, int):
            self.all_categories = False
            self.category_id = category
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __getitem__(self, index):
        """
        return:
            label_tensor of shape nx5, where n is number of labels in the image and x1,y1,x2,y2, and class_id.
        """
        #image, targets = super(CocoDetectionBoundingBox, self).__getitem__(index)
        img_id = self.ids[index]
        image = self._read_image(img_id)
        boxes, labels, _ = self._get_annotation(img_id) 

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels, img_id

    def get_image(self, index):
        image_id = self.ids[index]
        boxes, labels, _ = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, _, _ = self.transform(image, boxes, labels)
        return image

    def _read_image(self, img_id):
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image_file = os.path.join(self.root, path)
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return str(image_id), self._get_annotation(image_id)

    def _get_annotation(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        targets = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        is_difficult = []
        for target in targets:
            bbox = target["bbox"]  # in xywh format
            category_id = target["category_id"]
            if (not self.all_categories) and (category_id != self.category_id):
                continue
            category_id = self._delete_coco_empty_category(category_id)
            labels.append(category_id + 1)
            boxes.append(bbox)
            is_difficult.append(0)
        labels = np.array(labels, dtype=np.int64)
        is_difficult = np.array(is_difficult, dtype=np.int64)
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes) > 0:
            boxes = xywh_to_xyxy(boxes)
        else:
            boxes = np.zeros((1, 4), dtype=np.float32)
            #boxes = np.array([[1, 1, 100, 100]], dtype=np.float32)
            labels = np.zeros((1,), dtype=np.int64)
            is_difficult = np.zeros((1,), dtype=np.int64)

        return (boxes, labels, is_difficult)

    def _delete_coco_empty_category(self, old_id):
        """The COCO dataset has 91 categories but 11 of them are empty.
        This function will convert the 80 existing classes into range [0-79].
        Note the COCO original class index starts from 1.
        The converted index starts from 0.
        Args:
            old_id (int): The category ID from COCO dataset.
        Return:
            new_id (int): The new ID after empty categories are removed."""
        starting_idx = 1
        new_id = old_id - starting_idx
        for missing_id in self.missing_ids:
            if old_id > missing_id:
                new_id -= 1
            elif old_id == missing_id:
                raise KeyError(
                    "illegal category ID in coco dataset! ID # is {}".format(old_id)
                )
            else:
                break
        return new_id

    def add_coco_empty_category(self, old_id):
        """The reverse of delete_coco_empty_category."""
        starting_idx = 1
        new_id = old_id + starting_idx
        for missing_id in self.missing_ids:
            if new_id >= missing_id:
                new_id += 1
            else:
                break
        return new_id
