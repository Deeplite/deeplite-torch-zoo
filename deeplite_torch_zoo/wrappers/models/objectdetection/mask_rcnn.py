
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["fasterrcnn_resnet50_fpn_coco_80"]


@MODEL_WRAPPER_REGISTRY.register('fasterrcnn_resnet50_fpn', 'coco_80')
def fasterrcnn_resnet50_fpn_coco_80(pretrained=False, num_classes=91, progress=True, device="cuda"):
    return fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes)
