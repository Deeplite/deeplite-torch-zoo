
import os

from os.path import expanduser
from pathlib import Path

from deeplite_torch_zoo.src.objectdetection.ssd.eval import ssd_eval
from deeplite_torch_zoo.src.objectdetection.eval.coco.coco_evaluator import ssd_eval_coco

from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.vgg_ssd import create_vgg_ssd_predictor
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd_predictor
from deeplite_torch_zoo.src.objectdetection.ssd.models.mobilenetv2_ssd import create_mobilenetv2_ssd_predictor
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_predictor
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY


__all__ = [
    "vgg16_ssd_eval_func",
    "mb1_ssd_eval_func",
    "mb2_ssd_eval_func",
    "mb2_ssd_lite_eval_func",
    "mb2_ssd_coco_eval_func",
]


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='ssd', dataset_type='wider_face')
@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='ssd', dataset_type='voc')
@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='vgg16_ssd', dataset_type='voc')
@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='vgg16_ssd', dataset_type='wider_face')
@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='resnet18_ssd', dataset_type='voc')
@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='resnet34_ssd', dataset_type='voc')
@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='resnet50_ssd', dataset_type='voc')
def vgg16_ssd_eval_func(
    model, data_loader, iou_threshold=0.5, use_2007_metric=True, device="cuda"
):
    eval_path = os.path.join(expanduser("~"), ".deeplite-torch-zoo/voc/eval_results")
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    model.is_test = True
    predictor = create_vgg_ssd_predictor(model, nms_method="hard", device=device)

    stat = ssd_eval(
        predictor=predictor,
        dataloader=data_loader,
        data_path=eval_path,
        class_names=data_loader.dataset.classes,
        iou_threshold=iou_threshold,
        use_2007_metric=use_2007_metric,
    )
    model.is_test = False

    return stat


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='mb1_ssd', dataset_type='voc')
def mb1_ssd_eval_func(
    model, data_loader, iou_threshold=0.5, use_2007_metric=True, device="cuda"
):
    eval_path = os.path.join(expanduser("~"), ".deeplite-torch-zoo/voc/eval_results")
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    model.is_test = True
    predictor = create_mobilenetv1_ssd_predictor(
        model, nms_method="hard", device=device
    )

    stat = ssd_eval(
        predictor=predictor,
        dataloader=data_loader,
        data_path=eval_path,
        class_names=data_loader.dataset.classes,
        iou_threshold=iou_threshold,
        use_2007_metric=use_2007_metric,
    )

    model.is_test = False

    return stat


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='mb2_ssd', dataset_type='voc')
def mb2_ssd_eval_func(
    model, data_loader, iou_threshold=0.5, use_2007_metric=True, device="cuda"
):
    eval_path = os.path.join(expanduser("~"), ".deeplite-torch-zoo/voc/eval_results")
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    model.is_test = True
    predictor = create_mobilenetv2_ssd_predictor(
        model, nms_method="hard", device=device
    )

    stat = ssd_eval(
        predictor=predictor,
        dataloader=data_loader,
        data_path=eval_path,
        class_names=data_loader.dataset.classes,
        iou_threshold=iou_threshold,
        use_2007_metric=use_2007_metric,
    )

    model.is_test = False

    return stat


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='mb2_ssd', dataset_type='coco')
def mb2_ssd_coco_eval_func(
    model, data_loader, gt=None, iou_threshold=0.5, use_2007_metric=True, _set="voc", device="cuda"
):
    eval_path = os.path.join(expanduser("~"), f".deeplite-torch-zoo/{_set}/eval_results")
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    model.is_test = True
    predictor = create_mobilenetv2_ssd_predictor(
        model, nms_method="hard", device=device
    )
    stat = ssd_eval_coco(
        model,
        data_loader,
        gt=gt,
        predictor=predictor,
    )
    model.is_test = False

    return stat


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='mb2_ssd_lite', dataset_type='voc')
def mb2_ssd_lite_eval_func(
    model, data_loader, iou_threshold=0.5, use_2007_metric=True, device="cuda"
):
    eval_path = os.path.join(expanduser("~"), ".deeplite-torch-zoo/voc/eval_results")
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    model.is_test = True
    predictor = create_mobilenetv2_ssd_lite_predictor(
        model, nms_method="hard", device=device
    )
    stat = ssd_eval(
        predictor=predictor,
        dataloader=data_loader,
        data_path=eval_path,
        class_names=data_loader.dataset.classes,
        iou_threshold=iou_threshold,
        use_2007_metric=use_2007_metric,
    )

    model.is_test = False

    return stat
