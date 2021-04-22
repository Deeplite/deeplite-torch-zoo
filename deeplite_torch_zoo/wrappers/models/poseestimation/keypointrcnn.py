
from torchvision.models.detection import keypointrcnn_resnet50_fpn


__all__ = ["keypointrcnn_resnet50_fpn_coco_80"]


def keypointrcnn_resnet50_fpn_coco_80(pretrained=False, progress=True, device="cuda"):
    return keypointrcnn_resnet50_fpn(pretrained=pretrained, progress=progress).to(device)
