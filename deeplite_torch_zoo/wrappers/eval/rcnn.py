import torch

from deeplite_torch_zoo.src.objectdetection.eval.coco.mask_rcnn import RCNNCOCOEvaluator
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY


__all__ = ["rcnn_eval_coco"]


@EVAL_WRAPPER_REGISTRY.register('object_detection','fasterrcnn_resnet50_fpn','coco_80')
def rcnn_eval_coco(model, data_loader, gt=None, device="cuda", net="rcnn"):
    model.to(device)
    with torch.no_grad():
        return RCNNCOCOEvaluator(
            model,
            data_loader.dataset,
            gt=gt,
            net=net
        ).evaluate()
