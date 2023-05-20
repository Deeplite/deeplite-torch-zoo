import numpy as np

from deeplite_torch_zoo.src.objectdetection.eval.yolov5_eval.yolov5_eval import \
    evaluate
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='yolo', dataset_type='voc07')
@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='yolo', dataset_type='voc')
def yolo_eval_voc(
    model,
    test_dataloader,
    device="cuda",
    iou_thresh=0.5,
    conf_thresh=0.001,
    nms_thresh=0.5,
    eval_style='coco',
    progressbar=False,
    subclasses=None,
    num_classes=None,
    v8_eval=False,
    **kwargs
):
    model.to(device)
    ap_dict = evaluate(
        model,
        test_dataloader,
        conf_thres=conf_thresh,
        iou_thres=nms_thresh,
        eval_style=eval_style,
        map_iou_thresh=iou_thresh,
        num_classes=num_classes,
        v8_eval=v8_eval,
    )
    return ap_dict


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection', model_type='yolo', dataset_type='coco')
def yolo_eval_coco(
    model,
    test_dataloader,
    device="cuda",
    conf_thresh=0.001,
    nms_thresh=0.5,
    eval_style='coco',
    progressbar=False,
    subclasses=None,
    num_classes=None,
    **kwargs
):
    model.to(device)
    ap_dict = evaluate(
        model,
        test_dataloader,
        conf_thres=conf_thresh,
        iou_thres=nms_thresh,
        eval_style=eval_style,
        map_iou_thresh=np.arange(0.5, 1.0, 0.05),
        num_classes=num_classes,
    )
    return ap_dict
