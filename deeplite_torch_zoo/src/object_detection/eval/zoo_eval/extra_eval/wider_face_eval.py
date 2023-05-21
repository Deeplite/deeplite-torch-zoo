# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import cv2
import numpy as np
import torch

from deeplite_torch_zoo.src.object_detection.datasets.wider_face import WiderFace
from deeplite_torch_zoo.src.object_detection.eval.zoo_eval.evaluator import Evaluator
from deeplite_torch_zoo.src.object_detection.eval.zoo_eval.metrics import MAP
from deeplite_torch_zoo.api.registries import EVAL_WRAPPER_REGISTRY
from deeplite_torch_zoo.utils import LOGGER


class WiderFaceEval(Evaluator):
    """docstring for WiderFaceEval"""

    def __init__(self, model, data_root, net="yolov3", img_size=448):
        super(WiderFaceEval, self).__init__(model=model, img_size=img_size)

        self.dataset = WiderFace(data_root, split="val")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        self.model.eval()
        self.model.cuda()
        results = []
        start = time.time()
        for img_idx, _ in enumerate(self.dataset.img_info):
            _info = self.dataset.img_info[img_idx]
            img_path = _info["img_path"]
            image = cv2.imread(img_path)

            LOGGER.info(
                "Parsing batch: {}/{}".format(img_idx, len(self.dataset)), end="\r"
            )
            bboxes_prd = self.get_bbox(image)
            if len(bboxes_prd) == 0:
                bboxes_prd = np.zeros((0, 6))

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            detections = {"bboxes": [], "labels": []}
            detections["bboxes"] = bboxes_prd[:, :5]
            detections["labels"] = bboxes_prd[:, 5]

            gt_bboxes = _info["annotations"]["bbox"]
            gt_labels = np.ones(len(gt_bboxes))
            gt = {"bboxes": gt_bboxes, "labels": gt_labels}
            results.append({"detections": detections, "gt": gt})
        # put your model in training mode back on

        mAP = MAP(results, self.dataset.num_classes)
        mAP.evaluate()
        ap = mAP.accumlate()
        ap = ap[ap > 1e-6]
        _ap = np.mean(ap)
        LOGGER.info("mAP = {:.3f}".format(_ap))

        return _ap  # Average Precision  (AP) @[ IoU=050 ]


@EVAL_WRAPPER_REGISTRY.register(
    task_type='object_detection', model_type='yolo', dataset_type='wider_face'
)
def yolo_eval_wider_face(
    model, data_root, device="cuda", net="yolov3", img_size=448, **kwargs
):
    mAP = 0
    result = {}
    model.to(device)
    with torch.no_grad():
        mAP = WiderFaceEval(model, data_root, net=net, img_size=img_size).evaluate()
        result["mAP"] = mAP

    return result
