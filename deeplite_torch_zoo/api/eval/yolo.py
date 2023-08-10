from deeplite_torch_zoo.src.object_detection.eval.evaluate import evaluate
from deeplite_torch_zoo.api.registries import EVAL_WRAPPER_REGISTRY


@EVAL_WRAPPER_REGISTRY.register(task_type='object_detection')
def evaluate_detector(
    model,
    test_dataloader,
    device='cuda',
    iou_thres=0.6,
    conf_thres=0.001,
    num_classes=80,
    v8_eval=False,
    max_det=300,
    single_cls=False,
    augment=False,
    half=True,
    compute_loss=None,
):
    model.to(device)
    ap_dict = evaluate(
        model,
        test_dataloader,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        num_classes=num_classes,
        v8_eval=v8_eval,
        max_det=max_det,
        single_cls=single_cls,
        augment=augment,
        half=half,
        compute_loss=compute_loss,
    )
    return ap_dict
