import unittest
import pytest

from deeplite_torch_zoo import get_eval_function, list_models
from deeplite_torch_zoo.wrappers.eval import *
from deeplite_torch_zoo.src.segmentation.deeplab.eval import evaluate_deeplab
from deeplite_torch_zoo.src.segmentation.fcn.eval import evaluate_fcn
from deeplite_torch_zoo.src.segmentation.Unet.eval import eval_net, eval_net_miou
from deeplite_torch_zoo.src.objectdetection.eval.voc.voc_evaluator import (
    yolo_voc07_eval,
)
from deeplite_torch_zoo.wrappers.eval import mb2_ssd_coco_eval_func


classification_eval_list = [classification_eval]

objectdetection_eval_list = [
    rcnn_eval_coco,
    mb1_ssd_eval_func,
    mb2_ssd_coco_eval_func,
    mb2_ssd_lite_eval_func,
    mb2_ssd_lite_eval_func,
    mb2_ssd_eval_func,
    vgg16_ssd_eval_func,
    yolo_voc07_eval,
    yolo_eval_lisa,
    yolo_eval_voc,
    yolo_eval_coco,
    yolo_eval_wider_face,
]

segmentation_eval_list = [evaluate_fcn, eval_net, eval_net_miou, evaluate_deeplab]


class TestEvalFunction(unittest.TestCase):
    def test_eval_classification(self):
        all_classification_models = list_models(
            task_type_filter="classification", print_table=False, return_list=True
        )

        for (model_name, dataset_name) in all_classification_models:
            funct = get_eval_function(model_name=model_name, dataset_name=dataset_name)
            assert funct in classification_eval_list

    def test_eval_segmentation(self):
        all_segmentation_models = list_models(
            task_type_filter="semantic_segmentation",
            print_table=False,
            return_list=True,
        )

        for (model_name, dataset_name) in all_segmentation_models:

            funct = get_eval_function(model_name=model_name, dataset_name=dataset_name)
            assert funct in segmentation_eval_list

    def test_eval_objectdetection(self):
        all_objectdetection_models = list_models(
            task_type_filter="object_detection", print_table=False, return_list=True
        )

        for (model_name, dataset_name) in all_objectdetection_models:
            funct = get_eval_function(model_name=model_name, dataset_name=dataset_name)
            assert funct in objectdetection_eval_list


if __name__ == "__main__":
    unittest.main()
