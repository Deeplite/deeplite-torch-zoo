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
import random
import cv2

from PIL import ImageFile
from PIL import Image
from torchvision.datasets import CocoDetection

from deeplite_torch_zoo.src.objectdetection.eval.coco.utils import xywh_to_xyxy

class CocoDetectionBoundingBox(CocoDetection):
    def __init__(
        self,
        img_root,
        ann_file_name,
        num_classes=80,
        transform=None,
        category="all",
        img_size=416,
        classes=[],
        missing_ids=[]
    ):
        super(CocoDetectionBoundingBox, self).__init__(img_root, ann_file_name)
        self._tf = transform
        self._img_size = img_size
        self.classes = ["BACKGROUND"] + classes
        self.num_classes = len(self.classes)
        self.missing_ids = missing_ids
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
            label_tensor of shape nx6, where n is number of labels in the image and x1,y1,x2,y2, class_id and confidence.
        """
        img, targets = super(CocoDetectionBoundingBox, self).__getitem__(index)
        labels = []
        for target in targets:
            bbox = torch.tensor(target["bbox"], dtype=torch.float32)  # in xywh format
            category_id = target["category_id"]
            if (not self.all_categories) and (category_id != self.category_id):
                continue
            conf = torch.tensor([1.0])
            category_id = self._delete_coco_empty_category(category_id)
            category_id = torch.tensor([float(category_id)])
            label = torch.cat((bbox, category_id, conf))
            labels.append(label)
        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((0, 6))
        del labels

        if self._tf == None:
            return np.array(img), None, None, self.ids[index]
        transformed_img_tensor, label_tensor = self._tf(self._img_size)(
            img, label_tensor
        )
        label_tensor = xywh_to_xyxy(label_tensor)
        return (
            transformed_img_tensor,
            label_tensor,
            label_tensor.size(0),
            self.ids[index],
        )


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
            max_num_obj = max(max_num_obj, length)
            img_ids.append(torch.tensor([img_id]))
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
