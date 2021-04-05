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
from PIL import ImageFile
from torchvision.datasets import CocoDetection

from deeplite_torch_zoo.src.objectdetection.datasets.transforms import (default_transform_fn,
                                                    random_transform_fn)
from deeplite_torch_zoo.src.objectdetection.configs.coco_config import MISSING_IDS
from deeplite_torch_zoo.src.objectdetection.eval.coco.utils import xywh_to_xyxy

__all__ = ["get_coco_dataset"]


def get_coco_dataset(
    data_root, net, num_classes=80, num_torch_workers=1, batch_size=32, img_size=416
):
    if "yolo" in net:
        return get_coco_dataset_for_yolo(
            data_root,
            num_classes=num_classes,
            num_torch_workers=num_torch_workers,
            batch_size=batch_size,
            img_size=img_size,
        )
    else:
        raise ValueError


def get_coco_dataset_for_yolo(
    data_root, num_classes=80, num_torch_workers=1, batch_size=32, img_size=416
):

    train_trans = random_transform_fn

    train_annotate = os.path.join(data_root, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(data_root, "images/train2017")
    train_coco = CocoDetectionBoundingBox(
        train_coco_root,
        train_annotate,
        num_classes=num_classes,
        transform=train_trans,
        img_size=img_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_coco,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_torch_workers,
        collate_fn=train_coco.collate_img_label_fn,
    )

    val_annotate = os.path.join(data_root, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(data_root, "images/val2017")
    val_coco = CocoDetectionBoundingBox(
        val_coco_root, val_annotate, num_classes=num_classes, img_size=img_size
    )

    val_loader = torch.utils.data.DataLoader(
        val_coco,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_torch_workers,
        collate_fn=val_coco.collate_img_label_fn,
    )

    return {"train": train_loader, "val": val_loader}


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
            category_id = _delete_coco_empty_category(category_id)
            category_id = torch.tensor([float(category_id)])
            label = torch.cat((bbox, category_id, conf))
            labels.append(label)
        if labels:
            label_tensor = torch.stack(labels)
        else:
            label_tensor = torch.zeros((0, 6))
        del labels

        if self._tf == None:
            return np.array(img), self.ids[index]
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
