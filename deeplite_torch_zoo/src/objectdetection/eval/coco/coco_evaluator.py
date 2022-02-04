import os
import json
import time
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from deeplite_torch_zoo.src.objectdetection.eval.evaluator import Evaluator
from deeplite_torch_zoo.src.objectdetection.datasets.coco import CocoDetectionBoundingBox
from deeplite_torch_zoo.src.objectdetection.configs.coco_config import COCO_MISSING_IDS, COCO_DATA_CATEGORIES


class SubsampledCOCO(COCO):
    def __init__(self, annotation_file=None, subsample_categories=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.dataset['categories'] = [cat for cat in self.dataset['categories']
                if cat['name'] in subsample_categories]
            self.createIndex()


class COCOEvaluator(Evaluator):
    def __init__(
        self,
        model,
        dataset,
        visualize=False,
        net="yolo3",
        img_size=448,
        gt=None,
    ):
        data_path = "results/coco/{net}/".format(net=net)
        super(COCOEvaluator, self).__init__(
            model=model, data_path=data_path, net=net, img_size=img_size
        )
        self.dataset = dataset

        self.classes = self.dataset.classes
        self.__visiual = visualize
        self.__visual_imgs = 0

        self.cocoGt = gt

    def evaluate(self, multi_test=False, flip_test=False):
        results = []
        for img, _, _, img_ind in tqdm(self.dataset):
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
    def __init__(self, model, dataset, gt=None, visualize=False, net="yolo3", img_size=448):
        super().__init__(
            model=model,
            dataset=dataset,
            gt=gt,
            visualize=visualize,
            net=net,
            img_size=img_size
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


def yolo_eval_coco(model, data_root, gt=None, device="cuda",
                   net="yolo3", img_size=448, subsample_category=None, **kwargs):
    val_annotate = os.path.join(data_root, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(data_root, "val2017")

    if subsample_category is not None:
        if subsample_category in COCO_DATA_CATEGORIES["CLASSES"]:
            categories = [subsample_category, ]
            category_flag = COCO_DATA_CATEGORIES["CLASSES"].index(subsample_category)
            missing_ids = [category for category in list(range(1, 92)) if category != category_flag]
        else:
            raise RuntimeError(f'Category {subsample_category} is not present in the COCO dataset')
    else:
        categories = COCO_DATA_CATEGORIES["CLASSES"]
        category_flag = 'all'
        missing_ids = COCO_MISSING_IDS

    dataset = CocoDetectionBoundingBox(val_coco_root, val_annotate,
        classes=categories, category=category_flag, missing_ids=missing_ids)

    model.to(device)
    with torch.no_grad():
        return YoloCOCOEvaluator(model, dataset, gt=gt, net=net, img_size=img_size).evaluate()
