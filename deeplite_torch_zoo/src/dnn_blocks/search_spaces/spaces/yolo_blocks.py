from deeplite_torch_zoo.src.dnn_blocks.yolov7.yolo_blocks import (
    YOLOBottleneck,
    YOLOBottleneckCSP,
    YOLOBottleneckCSP2,
)
from deeplite_torch_zoo.src.dnn_blocks.yolov8.yolo_ultralytics_blocks import YOLOC3

from deeplite_torch_zoo.src.dnn_blocks.search_spaces.block_registry import (
    DNNBlockRegistry,
)

EXPANSION_FACTOR_RANGE = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


YOLO_BLOCK_REGISTRY = DNNBlockRegistry('yolo')


YOLO_BLOCK_REGISTRY.register(
    name='YOLOBottleneck',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
)(YOLOBottleneck)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOBottleneckCSP',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    shortcut=(False, True),
)(YOLOBottleneckCSP)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOBottleneckCSP2',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    shortcut=(False, True),
)(YOLOBottleneckCSP2)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC3',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    shortcut=(False, True),
)(YOLOC3)
