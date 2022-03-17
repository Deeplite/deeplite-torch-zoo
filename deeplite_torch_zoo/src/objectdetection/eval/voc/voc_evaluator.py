import os
import shutil
from os.path import expanduser
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import torch
from functools import partial

import deeplite_torch_zoo.src.objectdetection.configs.hyps.hyp_config_voc as cfg
from deeplite_torch_zoo.src.objectdetection.eval.evaluator import Evaluator
from deeplite_torch_zoo.src.objectdetection.eval.voc import voc_eval
from deeplite_torch_zoo.src.objectdetection.yolov5.utils.visualize import visualize_boxes

from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY


class VOCEvaluator(Evaluator):
    def __init__(
        self,
        model,
        voc2007_data_root,
        num_classes=20,
        visiual=False,
        net="yolo3",
        img_size=448,
        is_07_subset=False,
        progressbar=False,
    ):
        self.is_07_subset = is_07_subset
        self.test_file = "test.txt" if not self.is_07_subset else "val.txt"
        data_path = "deeplite_torch_zoo/results/voc/{net}".format(net=net)

        super(VOCEvaluator, self).__init__(
            model=model, data_path=data_path, img_size=img_size, net=net
        )

        self.progressbar = progressbar
        self.classes = cfg.DATA["CLASSES"]
        self.all_classes = cfg.DATA["ALLCLASSES"]
        if num_classes == 1:
            self.classes = cfg.DATA["CLASSES_1"]
        elif num_classes == 2:
            self.classes = cfg.DATA["CLASSES_2"]
        elif num_classes == 3:
            self.classes = cfg.DATA["CLASSES_3"]
<<<<<<< HEAD
<<<<<<< HEAD

        self.all_classes = cfg.DATA["ALLCLASSES"]
=======
=======
>>>>>>> 10b5313e1fc35f575e1162b5725d922851f1e3e1
        elif num_classes == 8:
            self.classes = cfg.DATA["CLASSES_8"]
            self.all_classes = cfg.DATA["CLASSES_8"]
        
        
<<<<<<< HEAD
>>>>>>> 10b5313e1fc35f575e1162b5725d922851f1e3e1
=======
>>>>>>> 10b5313e1fc35f575e1162b5725d922851f1e3e1
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}

        self.class_to_id_all = dict(zip(self.all_classes, range(len(self.all_classes))))

        self.map_selected_ids_to_all = {
            k: self.class_to_id_all[v] for k, v in self.id_to_class.items()
        }

        self.map_all_ids_to_selected = {
            v: k for k, v in self.map_selected_ids_to_all.items()
        }
        self.__visiual = visiual
        self.__visual_imgs = 0
        self.val_data_path = voc2007_data_root
        Path(data_path).mkdir(parents=True, exist_ok=True)

    def evaluate(self, multi_test=False, flip_test=False):
        img_inds_file = os.path.join(
            self.val_data_path, "ImageSets", "Main", self.test_file
        )
        with open(img_inds_file, "r") as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.mkdir(self.pred_result_path)

        for img_ind in tqdm(img_inds, disable=not self.progressbar):
            img_path = os.path.join(self.val_data_path, "JPEGImages", img_ind + ".jpg")
            img = cv2.imread(img_path)
            self.process_image(
                img, img_ind=img_ind, multi_test=multi_test, flip_test=flip_test
            )

        return self.__calc_APs()

    def process_image(self, img, img_ind, multi_test=False, flip_test=False):

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
                cfg.PROJECT_PATH, "data/results/{}.jpg".format(self.__visual_imgs)
            )
            cv2.imwrite(path, img)

            self.__visual_imgs += 1
        for bbox in bboxes_prd:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            class_name = self.id_to_class[class_ind]

            class_ind = self.map_selected_ids_to_all[int(bbox[5])]
            score = "%.4f" % score
            xmin, ymin, xmax, ymax = map(str, coor)
            s = " ".join([img_ind, score, xmin, ymin, xmax, ymax]) + "\n"

            with open(
                os.path.join(
                    self.pred_result_path, "comp4_det_test_" + class_name + ".txt"
                ),
                "a",
            ) as f:
                f.write(s)

    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        filename = os.path.join(self.pred_result_path, "comp4_det_test_{:s}.txt")
        cachedir = os.path.join(self.pred_result_path, "cache")
        annopath = os.path.join(self.val_data_path, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(self.val_data_path, "ImageSets", "Main", self.test_file)
        APs = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                iou_thresh,
                use_07_metric,
            )
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs




@EVAL_WRAPPER_REGISTRY.register('object_detection_yolo_voc')
def yolo_eval_voc(
    model, data_root, num_classes=20, device="cuda", net="yolo3",
    img_size=448, is_07_subset=False, progressbar=False, **kwargs
):

    mAP = 0
    result = {}
    model.to(device)
    with torch.no_grad():
        APs = VOCEvaluator(
            model, data_root, num_classes=num_classes, net=net,
            img_size=img_size, is_07_subset=is_07_subset, progressbar=progressbar,
        ).evaluate()
        for i in APs:
            # print("{} --> mAP : {}".format(i, APs[i]))
            result[i] = APs[i]
            mAP += APs[i]
        mAP = mAP / len(APs)
        # print('mAP:%g' % (mAP))
        result["mAP"] = mAP

    return result


@EVAL_WRAPPER_REGISTRY.register('object_detection_yolo_voc07')
def yolo_voc07_eval(
    model, data_root, num_classes=20, device="cuda", net="yolo3",
    img_size=448, is_07_subset=True, progressbar=False, **kwargs
):
    return yolo_eval_voc(model, data_root, num_classes=20, device="cuda", net="yolo3",
                img_size=448, is_07_subset=True, progressbar=False, **kwargs)

