import os

from tqdm import tqdm
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from deeplite_torch_zoo.src.objectdetection.eval.evaluator import Evaluator
from deeplite_torch_zoo.src.objectdetection.datasets.coco import CocoDetectionBoundingBox
from deeplite_torch_zoo.src.objectdetection.datasets.coco_config import COCO_MISSING_IDS, COCO_DATA_CATEGORIES
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY


class COCOEvaluator(Evaluator):
    def __init__(
        self,
        model,
        dataset,
        visualize=False,
        net="yolo3",
        img_size=448,
        gt=None,
        progressbar=False,
    ):
        data_path = "results/coco/{net}/".format(net=net)
        super(COCOEvaluator, self).__init__(
            model=model, data_path=data_path, net=net, img_size=img_size
        )
        self.dataset = dataset
        self.progressbar = progressbar

        self.classes = self.dataset.classes
        self.__visiual = visualize
        self.__visual_imgs = 0

        self.cocoGt = gt

    def evaluate(self, multi_test=False, flip_test=False):
        results = []
        for img, _, _, img_ind in tqdm(self.dataset, disable=not self.progressbar):
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
    def __init__(self, model, dataset, gt=None, visualize=False,
        net="yolo3", img_size=448, progressbar=False):
        super().__init__(
            model=model,
            dataset=dataset,
            gt=gt,
            visualize=visualize,
            net=net,
            img_size=img_size,
            progressbar=progressbar,
        )

    def process_image(self, img, img_ind, multi_test=False, flip_test=False, **kwargs):

        bboxes_prd = self.get_bbox(img, multi_test, flip_test)

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
    model.to(device)
    with torch.no_grad():
        return SSDCOCOEvaluator(
            model,
            data_loader.dataset,
            gt=gt,
            predictor=predictor,
            net=net
        ).evaluate()


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='yolo', dataset_type='coco')
def yolo_eval_coco(model, data_root, gt=None, device="cuda",
                   net="yolo3", img_size=448, subsample_categories=None, progressbar=False, **kwargs):

    val_annotate = os.path.join(data_root, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(data_root, "val2017")

    if subsample_categories is not None:
        categories = subsample_categories
        category_indices = [COCO_DATA_CATEGORIES["CLASSES"].index(cat) + 1 for cat in categories]
        missing_ids = [category for category in list(range(1, 92)) if category not in category_indices]
    else:
        categories = COCO_DATA_CATEGORIES["CLASSES"]
        category_indices = 'all'
        missing_ids = COCO_MISSING_IDS

    dataset = CocoDetectionBoundingBox(val_coco_root, val_annotate,
        img_size=img_size, classes=categories, category=category_indices, missing_ids=missing_ids)

    model.to(device)
    with torch.no_grad():
        return YoloCOCOEvaluator(model, dataset, gt=gt, net=net,
            img_size=img_size, progressbar=progressbar).evaluate()


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='yolo', dataset_type='car_detection')
def yolo_eval_coco_car(model, data_root, gt=None, device="cuda",
    net="yolo3", img_size=448, subsample_categories=["car"], progressbar=False, **kwargs):
    return yolo_eval_coco(model, data_root, gt=None, device=device,
        net=net, img_size=img_size, subsample_categories=subsample_categories,
        progressbar=progressbar, **kwargs)
