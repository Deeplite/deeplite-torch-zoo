from functools import partial

from deeplite_torch_zoo.src.objectdetection.eval.coco.coco_evaluator import yolo_eval_coco
from deeplite_torch_zoo.src.objectdetection.eval.lisa_eval import yolo_eval_lisa
from deeplite_torch_zoo.src.objectdetection.eval.wider_face_eval import yolo_eval_wider_face
from deeplite_torch_zoo.src.objectdetection.eval.voc.voc_evaluator import yolo_eval_voc
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY

__all__ = ["yolo_eval_coco", "yolo_eval_voc",
    "yolo_eval_lisa", "yolo_eval_wider_face"]


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
for key, out_wrapper_fn in EVAL_FN_MAP.items():
    if (key == 'voc07'):
        @EVAL_WRAPPER_REGISTRY.register('object_detection','yolo','voc07')
        def wrapper_fn( model, data_root="", num_classes=20, device="cuda", net="yolo3",
                        img_size=448, is_07_subset=True, progressbar=False, **kwargs):
            return partial(out_wrapper_fn(model, 
                        data_root=data_root,
                        num_classes=num_classes,
                        device=device,
                        net=net,
                        img_size=img_size,
                        is_07_subset=is_07_subset, 
                        progressbar=progressbar))
    
    elif (key == 'voc'):
        @EVAL_WRAPPER_REGISTRY.register('object_detection','yolo','voc')
        def wrapper_fn(model, data_root="", num_classes=20, device="cuda", net="yolo3",
                        img_size=448, is_07_subset=False, progressbar=False, **kwargs):
            return out_wrapper_fn(model, 
                        data_root=data_root,
                        num_classes=num_classes,
                        device=device,
                        net=net,
                        img_size=img_size,
                        is_07_subset=is_07_subset, 
                        progressbar=progressbar)
    
    elif key == 'coco':

        @EVAL_WRAPPER_REGISTRY.register('object_detection','yolo','coco')
        def wrapper_fn(model, data_root="", gt=None, device="cuda",
            net="yolo3", img_size=448, subsample_categories=None, progressbar=False, **kwargs):
            return out_wrapper_fn(model, 
                        data_root=data_root,
                        gt=gt,
                        device=device,
                        net=net,
                        img_size=img_size,
                        subsample_categories=subsample_categories, 
                        progressbar=progressbar)
                    
    elif key == 'lisa':

        @EVAL_WRAPPER_REGISTRY.register('object_detection','yolo','lisa')
        def wrapper_fn(model, data_root="", device="cuda", net="yolov3", img_size=448, **kwargs):
            return out_wrapper_fn(model, 
                        data_root=data_root,
                        device=device,
                        net=net,
                        img_size=img_size,)

    elif key == 'wider_face':

        @EVAL_WRAPPER_REGISTRY.register('object_detection','yolo','wider_face')
        def wrapper_fn(model, data_root="", device="cuda", net="yolov3", img_size=448, **kwargs):
            return out_wrapper_fn(model, 
                        data_root=data_root,
                        device=device,
                        net=net,
                        img_size=img_size,
                        )
    
    elif key == 'car_detection':

        @EVAL_WRAPPER_REGISTRY.register('object_detection','yolo','car_detection')
        def wrapper_fn(model, data_root="", num_classes=20, device="cuda", net="yolo3",
                        img_size=448, is_07_subset=False, progressbar=False,subsample_categories=['car'], **kwargs):
            return partial(out_wrapper_fn(model, 
                        data_root=data_root,
                        num_classes=num_classes,
                        device=device,
                        net=net,
                        img_size=img_size,
                        is_07_subset=is_07_subset, 
                        progressbar=progressbar,
                        subsample_categories=subsample_categories,
                        ))

