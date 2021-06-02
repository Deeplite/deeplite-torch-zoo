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

from deeplite_torch_zoo.src.objectdetection.configs.coco_config import MISSING_IDS, DATA
from deeplite_torch_zoo.src.objectdetection.eval.coco.utils import xywh_to_xyxy
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.data_augment import Mixup

class CocoDetectionBoundingBox(CocoDetection):
    def __init__(
        self,
        img_root,
        ann_file_name,
        num_classes=80,
        transform=None,
        category="all",
        img_size=416,
    ):
        super(CocoDetectionBoundingBox, self).__init__(img_root, ann_file_name)
        self._tf = transform
        self.num_classes = num_classes
        self._img_size = img_size
        self.classes = ["BACKGROUND"] + DATA["CLASSES"]
        if category == "all":
            self.all_categories = True
            self.category_id = -1
        elif isinstance(category, int):
            self.all_categories = False
            self.category_id = category
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def _parse_targets(self, targets):
        labels = []
        for target in targets:
            bbox = target["bbox"]  # in xywh format
            category_id = target["category_id"]
            if (not self.all_categories) and (category_id != self.category_id):
                continue
            conf = [1.0]
            category_id = _delete_coco_empty_category(category_id)
            category_id = [float(category_id)]
            label = np.concatenate((bbox, category_id, conf), axis=0)
            labels.append(label)
        if labels:
            labels = np.array(labels)
        else:
            labels = np.zeros((0, 6))
        return labels

    def _read_data(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        labels = self._parse_targets(target)
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = cv2.imread(img_path)
        if self._tf == None:
            return np.array(img), self.ids[index]
        transformed_img_tensor, label_tensor = self._tf(self._img_size)(
            Image.fromarray(img), torch.from_numpy(labels).float()
        )
        return transformed_img_tensor, label_tensor

    def __getitem__(self, index):
        """
        return:
            label_tensor of shape nx6, where n is number of labels in the image and x1,y1,x2,y2, class_id and confidence.
        """
        img, labels = self._read_data(index)
        if self._tf is None:
            return img, labels

        item_mix = random.randint(0, len(self.ids) - 1)
        img_mix, labels_mix = self._read_data(item_mix)

        img, labels = Mixup()(np.array(img), labels, np.array(img_mix), labels_mix)
        label_tensor = torch.tensor(labels)
        label_tensor = xywh_to_xyxy(label_tensor)
        return (
            torch.from_numpy(img).float(),
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


def _delete_coco_empty_category(old_id):
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
    for missing_id in MISSING_IDS:
        if old_id > missing_id:
            new_id -= 1
        elif old_id == missing_id:
            raise KeyError(
                "illegal category ID in coco dataset! ID # is {}".format(old_id)
            )
        else:
            break
    return new_id


def add_coco_empty_category(old_id):
    """The reverse of delete_coco_empty_category."""
    starting_idx = 1
    new_id = old_id + starting_idx
    for missing_id in MISSING_IDS:
        if new_id >= missing_id:
            new_id += 1
        else:
            break
    return new_id
