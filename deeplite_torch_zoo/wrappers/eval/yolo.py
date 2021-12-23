
from deeplite_torch_zoo.src.objectdetection.eval.coco.coco_evaluator import yolo_eval_coco
from deeplite_torch_zoo.src.objectdetection.eval.lisa_eval import yolo_eval_lisa
from deeplite_torch_zoo.src.objectdetection.eval.wider_face_eval import yolo_eval_wider_face
from deeplite_torch_zoo.src.objectdetection.eval.voc.voc_evaluator import yolo_eval_voc


__all__ = ["get_eval_func", "yolo_eval_coco", "yolo_eval_voc", "yolo_eval_lisa", "yolo_eval_wider_face"]


def get_eval_func(_set):
    if "voc" in _set:
        return yolo_eval_voc
    if "coco" in _set:
        return yolo_eval_coco
    if "lisa" in _set:
        return yolo_eval_lisa
    if 'wider_face' in _set:
        return yolo_eval_wider_face
    if 'person_detection' in _set:
        return yolo_eval_voc

    raise ValueError
