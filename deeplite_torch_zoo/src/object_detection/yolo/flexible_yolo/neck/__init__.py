# Code credit: https://github.com/Bobo-y/flexible-yolov5
# The file is modified by Deeplite Inc. from the original implementation on Jan 4, 2023

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.v5fpn import YOLOv5FPN
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.v5pan import YOLOv5PAN

__all__ = ['build_neck']

NECK_MAP = {
    'FPN': YOLOv5FPN,
    'PAN': YOLOv5PAN,
}


def build_neck(neck_name, **kwargs):
    if neck_name not in NECK_MAP:
        raise ValueError(
            f'Neck {neck_name} not supported. Supported neck types: {NECK_MAP.keys()}'
        )
    neck = NECK_MAP[neck_name](**kwargs)
    return neck
