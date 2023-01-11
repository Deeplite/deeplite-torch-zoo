from .efficientnet import efficientnet
from .gnn import gnn
from .hrnet import hrnet
from .mobilenetv3 import MobileNetV3 as mobilenetv3
from .repvgg import repvgg
from .resnet import resnet
from .shufflenetv2 import shufflenetv2
from .swin_transformer import swin_transformer as swin
from .vgg import vgg
from .yolov5 import YOLOv5

__all__ = ['build_backbone']

BACKBONE_MAP = {
    'resnet': resnet,
    'shufflenetv2': shufflenetv2,
    'mobilenetv3': mobilenetv3,
    'YOLOv5': YOLOv5,
    'efficientnet': efficientnet,
    'hrnet': hrnet,
    'swin': swin,
    'vgg': vgg,
    'repvgg': repvgg,
    'gnn': gnn,
}


def build_backbone(backbone_name, **kwargs):
    if backbone_name not in BACKBONE_MAP:
        raise ValueError(f'Backbone {backbone_name} not supported. Supported backbones: {BACKBONE_MAP.keys()}')
    backbone = BACKBONE_MAP[backbone_name](**kwargs)
    return backbone
