
from torchvision.models.detection import fasterrcnn_resnet50_fpn


__all__ = ["fasterrcnn_resnet50_fpn_coco_80"]


def fasterrcnn_resnet50_fpn_coco_80(pretrained=False, progress=True, device="cuda"):
    return fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress)
