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

import glob
import time

import cv2
import numpy as np
import torch

import deeplite_torch_zoo.src.objectdetection.configs.lisa_config as lisa_cfg
from deeplite_torch_zoo.src.objectdetection.datasets.lisa import LISA
from deeplite_torch_zoo.src.objectdetection.eval.evaluator import Evaluator
from deeplite_torch_zoo.src.objectdetection.eval.metrics import MAP
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.tools import post_process
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.visualize import visualize_boxes


class Demo(Evaluator):
    """docstring for Demo"""

    def __init__(
        self,
        model,
        data_root="data/esmart/images",
        visiual=False,
        net="yolov3",
        img_size=448,
    ):
        data_path = "deeplite_torch_zoo/results/lisa/{net}".format(net=net)
        super(Demo, self).__init__(
            model=model, data_path=data_path, img_size=img_size, net=net
        )

        self.data_root = data_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = lisa_cfg.DATA["CLASSES"]
        self.filelist = glob.glob("{}/*.jpg".format(self.data_root))
        self.filelist = sorted(self.filelist)

    def process(self):
        self.model.eval()
        self.model.cuda()
        results = []
        start = time.time()
        avg_loss = 0
        iter_ = 1
        for img_idx, (img_path) in enumerate(self.filelist):

            image = cv2.imread(img_path)
            bboxes_prd = self.get_bbox(image)

            boxes = bboxes_prd[..., :4]
            class_inds = bboxes_prd[..., 5].astype(np.int32)
            scores = bboxes_prd[..., 4]

            visualize_boxes(
                image=image,
                boxes=boxes,
                labels=class_inds,
                probs=scores,
                class_labels=self.classes,
            )
            path = "data/results/{:05}.jpg".format(iter_)
            iter_ = iter_ + 1
            print(path)
            cv2.imwrite(path, image)


class LISAEval(Evaluator):
    """docstring for LISAEval"""

    def __init__(self, model, data_root, visiual=False, net="yolov3", img_size=448):
        data_path = "deeplite_torch_zoo/results/lisa/{net}".format(net=net)
        super(LISAEval, self).__init__(
            model=model, data_path=data_path, img_size=img_size, net=net
        )

        self.dataset = val_dataset = LISA(data_root, _set="valid")
        self.data_root = data_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):

        self.model.eval()
        self.model.cuda()
        results = []
        start = time.time()
        avg_loss = 0
        for img_idx, (img_path) in enumerate(self.dataset.images):

            image = cv2.imread("{}/{}".format(self.data_root, img_path))
            label = self.dataset.objects[img_idx]
            print("Parsing batch: {}/{}".format(img_idx, len(self.dataset)), end="\r")
            bboxes_prd = self.get_bbox(image)
            if len(bboxes_prd) == 0:
                bboxes_prd = np.zeros((0, 6))

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            detections = {"bboxes": [], "labels": []}
            detections["bboxes"] = bboxes_prd[:, :5]
            detections["labels"] = bboxes_prd[:, 5]

            gt_bboxes = np.array(label["boxes"], dtype=np.float64)
            gt_labels = [self.dataset.label_map[_l] for _l in label["labels"]]
            gt = {"bboxes": gt_bboxes, "labels": gt_labels}
            results.append({"detections": detections, "gt": gt})
        print("validation loss = {}".format(avg_loss))
        # put your model in training mode back on

        mAP = MAP(results, self.dataset.num_classes)
        mAP.evaluate()
        ap = mAP.accumlate()
        mAP_all_classes = np.mean(ap)

        for i in range(0, ap.shape[0]):
            print("{:_>25}: {:.3f}".format(self.dataset.inv_map[i], ap[i]))

        print("(All Classes) AP = {:.3f}".format(np.mean(ap)))

        ap = ap[ap > 1e-6]
        ap = np.mean(ap)
        print("(Selected) AP = {:.3f}".format(ap))

        return ap  # Average Precision  (AP) @[ IoU=050 ]


def yolo_eval_lisa(model, data_root, device="cuda", net="yolov3", img_size=448, **kwargs):

    mAP = 0
    result = {}
    model.to(device)
    with torch.no_grad():
        mAP = LISAEval(model, data_root, net=net, img_size=img_size).evaluate()
        result["mAP"] = mAP

    return result
