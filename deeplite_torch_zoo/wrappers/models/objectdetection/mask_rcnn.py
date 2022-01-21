
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from deeplite_torch_zoo.utils.registry import MODEL_WRAPPER_REGISTRY


__all__ = ["fasterrcnn_resnet50_fpn_coco_80"]


@MODEL_WRAPPER_REGISTRY.register('fasterrcnn_resnet50_fpn','coco_80','objectdetection')
def fasterrcnn_resnet50_fpn_coco_80(pretrained=False, progress=True, device="cuda"):
    return fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress)
