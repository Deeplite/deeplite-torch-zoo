from functools import partial

from deeplite_torch_zoo.src.objectdetection.eval.coco.coco_evaluator import yolo_eval_coco
from deeplite_torch_zoo.src.objectdetection.eval.lisa_eval import yolo_eval_lisa
from deeplite_torch_zoo.src.objectdetection.eval.wider_face_eval import yolo_eval_wider_face
from deeplite_torch_zoo.src.objectdetection.eval.voc.voc_evaluator import yolo_eval_voc


__all__ = ["get_eval_func", "yolo_eval_coco", "yolo_eval_voc",
    "yolo_eval_lisa", "yolo_eval_wider_face"]


def get_eval_func(dataset_name):
    EVAL_FN_MAP = {
        'voc07': partial(yolo_eval_voc, is_07_subset=True),
        'voc': yolo_eval_voc,
        'coco': yolo_eval_coco,
        'lisa': yolo_eval_lisa,
        'wider_face': yolo_eval_wider_face,
        'person_detection': yolo_eval_voc,
        'car_detection': partial(yolo_eval_coco, subsample_categories=['car']),
        'person_pet_vehicle_detection': yolo_eval_voc,
    }
    for key, wrapper_fn in EVAL_FN_MAP.items():
        if key in dataset_name:
            return wrapper_fn
    return None
