from deeplite_torch_zoo.src.objectdetection.eval.coco.coco_evaluator import \
    yolo_eval_coco
from deeplite_torch_zoo.src.objectdetection.eval.lisa_eval import yolo_eval_lisa
from deeplite_torch_zoo.src.objectdetection.eval.voc.voc_evaluator import \
    yolo_eval_voc


__all__ = ["yolo_eval_func"]


def yolo_eval_func(
    model,
    data_root,
    num_classes=20,
    device="cuda",
    _set="voc",
    net="yolov3",
    img_size=448,
):
    """
    :param model: A yolo model to be evaluated.
    :param data_root: The path to the root of the dataset.
    returns an evaluator for a yolo models.
    """
    if "voc" in _set:
        return yolo_eval_voc(
            model,
            data_root,
            num_classes=num_classes,
            device=device,
            net=net,
            img_size=img_size,
        )
    if "coco" in _set:
        return yolo_eval_coco(model, data_root, device, net)
    if "lisa" in _set:
        return yolo_eval_lisa(
            model, data_root=data_root, device=device, net=net, img_size=img_size
        )
    raise ValueError
