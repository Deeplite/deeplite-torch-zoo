from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.backbone.resnet import resnet
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.backbone.yolo import YOLOv5Backbone, YOLOv8Backbone


__all__ = ['build_backbone']

BACKBONE_MAP = {
    'resnet': resnet,
    'yolo5': YOLOv5Backbone,
    'yolo8': YOLOv8Backbone,
}


def build_backbone(backbone_name, **kwargs):
    if backbone_name not in BACKBONE_MAP:
        raise ValueError(
            f'Backbone {backbone_name} not supported. '
            f'Supported backbones: {BACKBONE_MAP.keys()}'
        )
    backbone = BACKBONE_MAP[backbone_name](**kwargs)
    return backbone
