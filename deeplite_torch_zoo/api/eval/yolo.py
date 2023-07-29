from deeplite_torch_zoo.src.object_detection.eval.evaluate import evaluate
from deeplite_torch_zoo.api.registries import EVAL_WRAPPER_REGISTRY


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection')
def evaluate_detector(
    model,
    test_dataloader,
    device='cuda',
    iou_thresh=0.6,
    conf_thresh=0.001,
    num_classes=80,
    v8_eval=False,
    **kwargs
):
    model.to(device)
    ap_dict = evaluate(
        model,
        test_dataloader,
        conf_thres=conf_thresh,
        iou_thres=iou_thresh,
        num_classes=num_classes,
        v8_eval=v8_eval,
        **kwargs,
    )
    return ap_dict
