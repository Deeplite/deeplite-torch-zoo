from deeplite_torch_zoo.src.object_detection.flexible_yolo.backbone.resnet import resnet


__all__ = ['build_backbone']

BACKBONE_MAP = {
    'resnet': resnet,
}


def build_backbone(backbone_name, **kwargs):
    if backbone_name not in BACKBONE_MAP:
        raise ValueError(
            f'Backbone {backbone_name} not supported. '
            f'Supported backbones: {BACKBONE_MAP.keys()}'
        )
    backbone = BACKBONE_MAP[backbone_name](**kwargs)
    return backbone
