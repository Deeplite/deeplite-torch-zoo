import os
import shutil
from os.path import expanduser
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

import deeplite_torch_zoo.src.objectdetection.configs.coco_config as cfg
from deeplite_torch_zoo.src.objectdetection.mb_ssd.datasets.coco import CocoDetectionBoundingBox

from deeplite_torch_zoo.src.objectdetection.eval.evaluator import Evaluator
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.data_augment import Resize
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.tools import nms
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.visualize import visualize_boxes


class COCOEvaluator(Evaluator):
    def __init__(
        self,
        model,
        dataset,
        visiual=False,
        net="yolov3",
        img_size=448,
        gt=None,
    ):
        data_path = "results/coco/{net}/".format(net=net)
        super(COCOEvaluator, self).__init__(
            model=model, data_path=data_path, net=net, img_size=img_size
        )
        self.dataset = dataset

        self.classes = self.dataset.classes[1:]
        self.__visiual = visiual
        self.__visual_imgs = 0

        self.cocoGt = gt

    def evaluate(self, multi_test=False, flip_test=False):
        results = []
        for img, _, _, img_ind in self.dataset:
            results += self.process_image(img, int(img_ind))

        results = np.array(results).astype(np.float32)
        if len(results) == 0:
            return {"mAP": 0}
        cocoDt = self.cocoGt.loadRes(results)
        E = COCOeval(self.cocoGt, cocoDt, iouType="bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f}".format(E.stats[0]))
        return {"mAP": E.stats[0]}

    def process_image(self, img, **kwargs):
        pass


class YoloCOCOEvaluator(COCOEvaluator):
    def __init__(self, model, dataset, visualize=False, net="yolov3", img_size=448):
        super().__init__(
            model=model,
            dataset=dataset,
            visualize=visualize,
            net=net,
            img_size=img_size
        )

    def process_image(self, img, img_ind, multi_test=False, flip_test=False, **kwargs):

        bboxes_prd = self.get_bbox(img, multi_test, flip_test)

        if bboxes_prd.shape[0] != 0 and self.__visiual and self.__visual_imgs < 100:
            boxes = bboxes_prd[..., :4]
            class_inds = bboxes_prd[..., 5].astype(np.int32)
            scores = bboxes_prd[..., 4]

            visualize_boxes(
                image=img,
                boxes=boxes,
                labels=class_inds,
                probs=scores,
                class_labels=self.classes,
            )
            path = os.path.join(
                "deeplite_torch_zoo/results/{}.jpg".format(self.__visual_imgs)
            )
            cv2.imwrite(path, img)

            self.__visual_imgs += 1
        results = []
        for bbox in bboxes_prd:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])

            class_name = self.classes[class_ind]
            xmin, ymin, xmax, ymax = coor
            results.append(
                [
                    img_ind,
                    xmin,
                    ymin,
                    xmax - xmin,
                    ymax - ymin,
                    score,
                    self.dataset.add_coco_empty_category(class_ind),
                ]
            )
        return results


class SSDCOCOEvaluator(COCOEvaluator):
    def __init__(self, model, dataset, gt=None, net="ssd", predictor=None, img_size=300):
        super().__init__(
            model=model,
            dataset=dataset,
            net=net,
            img_size=img_size,
            gt=gt,
        )
        self.predictor = predictor

    def process_image(self, img, img_id):
        boxes, labels, probs = self.predictor.predict(img)
        results = []
        for bbox, label, prob in zip(boxes, labels, probs):
            xmin, ymin, xmax, ymax = bbox
            results.append(
                [
                    img_id,
                    xmin,
                    ymin,
                    xmax - xmin,
                    ymax - ymin,
                    prob,
                    self.dataset.add_coco_empty_category(label) - 1,
                ]
            )
        return results


def ssd_eval_coco(model, data_loader, gt=None, predictor=None, device="cuda", net="ssd"):
    mAP = 0
    result = {}
    model.to(device)
    with torch.no_grad():
        return SSDCOCOEvaluator(
            model,
            data_loader.dataset,
            gt=gt,
            predictor=predictor,
            net=net
        ).evaluate()


def yolo_eval_coco(model, data_loader, device="cuda", net="yolov3"):
    mAP = 0
    result = {}
    model.to(device)
    with torch.no_grad():
        return YoloCOCOEvaluator(model, data_loader.dataset, net=net).evaluate()
