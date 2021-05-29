
from deeplite_torch_zoo.src.objectdetection.eval.coco.coco_evaluator import yolo_eval_coco
from deeplite_torch_zoo.src.objectdetection.eval.lisa_eval import yolo_eval_lisa
from deeplite_torch_zoo.src.objectdetection.eval.voc.voc_evaluator import yolo_eval_voc
from deeplite_torch_zoo.src.objectdetection.eval.nssol_eval import yolo_eval_nssol


__all__ = ["get_eval_func"]


def yolo_eval_func(
    model,
    data_root,
    data_loader=None,
    gt=None,
    num_classes=20,
    device="cuda",
    _set="voc",
    net="yolov3",
    img_size=448,
    **kwargs,
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
        return yolo_eval_coco(model, data_loader=data_loader, gt=gt, device=device, net=net)
    if "lisa" in _set:
        return yolo_eval_lisa(
            model, data_root=data_root, device=device, net=net, img_size=img_size
        )
    raise ValueError

def get_eval_func(_set):
    if "voc" in _set:
        return yolo_eval_voc
    if "coco" in _set:
        return yolo_eval_coco
    if "lisa" in _set:
        return yolo_eval_lisa
    if "nssol" in _set:
        return yolo_eval_nssol
    raise ValueError
