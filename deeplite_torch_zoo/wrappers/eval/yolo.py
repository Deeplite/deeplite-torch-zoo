
from deeplite_torch_zoo.src.objectdetection.eval.coco.coco_evaluator import yolo_eval_coco, yolo_eval_lego
from deeplite_torch_zoo.src.objectdetection.eval.lisa_eval import yolo_eval_lisa
from deeplite_torch_zoo.src.objectdetection.eval.voc.voc_evaluator import yolo_eval_voc
from deeplite_torch_zoo.src.objectdetection.eval.nssol_eval import yolo_eval_nssol


__all__ = ["get_eval_func"]


def get_eval_func(_set):
    if "voc" in _set:
        return yolo_eval_voc
    if "coco" in _set:
        return yolo_eval_coco
    if "lego" in _set:
        return yolo_eval_lego
    if "lisa" in _set:
        return yolo_eval_lisa
    if "nssol" in _set:
        return yolo_eval_nssol
    raise ValueError
