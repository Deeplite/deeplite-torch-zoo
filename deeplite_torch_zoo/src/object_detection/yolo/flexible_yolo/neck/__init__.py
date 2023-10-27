# Code credit: https://github.com/Bobo-y/flexible-yolov5
# The file is modified by Deeplite Inc. from the original implementation on Jan 4, 2023

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.v5fpn import YOLOv5FPN
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.v5pan import YOLOv5PAN
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.v8fpn import YOLOv8FPN
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.v8pan import YOLOv8PAN
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.mmyolo_necks import YOLOv7PAFPNWrapper, PPYOLOECSPPAFPNWrapper

__all__ = ['build_neck']

NECK_MAP = {
    'v5FPN': YOLOv5FPN,
    'v5PAN': YOLOv5PAN,
    'v8FPN': YOLOv8FPN,
    'v8PAN': YOLOv8PAN,
    'v7PAFPN': YOLOv7PAFPNWrapper,
    'PPYOLOECSPPAFPN': PPYOLOECSPPAFPNWrapper,
}


def build_neck(neck_name, **kwargs):
    if neck_name not in NECK_MAP:
        raise ValueError(
            f'Neck {neck_name} not supported. Supported neck types: {NECK_MAP.keys()}'
        )
    neck = NECK_MAP[neck_name](**kwargs)
    return neck
