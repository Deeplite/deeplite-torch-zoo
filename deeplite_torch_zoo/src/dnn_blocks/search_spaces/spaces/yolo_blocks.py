from deeplite_torch_zoo.src.dnn_blocks.yolov7.yolo_blocks import (
    YOLOBottleneck,
    YOLOBottleneckCSP,
    YOLOBottleneckCSP2,
)
from deeplite_torch_zoo.src.dnn_blocks.yolov8.yolo_ultralytics_blocks import YOLOC3, YOLOC2, YOLOC2f, YOLOC1, YOLOC3x

from deeplite_torch_zoo.src.dnn_blocks.search_spaces.block_registry import (
    DNNBlockRegistry,
)

EXPANSION_FACTOR_RANGE = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
IN_EXPANSION_FACTOR_RANGE = (0.75, 0.8, 0.9, 1.0)
SHORTCUT_FLAGS = (True, False)


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
    shortcut=SHORTCUT_FLAGS,
)(YOLOBottleneckCSP)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOBottleneckCSP2',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    shortcut=SHORTCUT_FLAGS,
)(YOLOBottleneckCSP2)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC3',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    shortcut=SHORTCUT_FLAGS,
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOC3)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC2',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    shortcut=SHORTCUT_FLAGS,
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOC2)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC2f',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    shortcut=SHORTCUT_FLAGS,
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOC2f)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC1',
    k=(3, 5),
    n=(1, 2, 3),
)(YOLOC1)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC3x',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    shortcut=SHORTCUT_FLAGS,
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOC3x)
