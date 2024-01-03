from deeplite_torch_zoo.src.dnn_blocks.yolov7.yolo_blocks import (
    YOLOBottleneck,
    YOLOBottleneckCSP,
    YOLOBottleneckCSP2,
    YOLOVoVCSP,
    YOLOBottleneckCSPF,
    YOLOBottleneckCSPL,
    YOLOBottleneckCSPLG,
    BottleneckCSPA,
    BottleneckCSPB,
    BottleneckCSPC,
    ResCSPA,
    ResCSPB,
    ResCSPC,
)
from deeplite_torch_zoo.src.dnn_blocks.yolov8.yolo_ultralytics_blocks import YOLOC3, YOLOC2, YOLOC2f, YOLOC1, YOLOC3x

from deeplite_torch_zoo.src.dnn_blocks.search_spaces.block_registry import (
    DNNBlockRegistry,
)


EXPANSION_FACTOR_RANGE = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
IN_EXPANSION_FACTOR_RANGE = (0.75, 0.8, 0.9, 1.0)


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
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOBottleneckCSP)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOBottleneckCSP2',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOBottleneckCSP2)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC3',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOC3)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC3x',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOC3x)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC2',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOC2)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC2f',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOC2f)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOC1',
    k=(3, 5),
    n=(1, 2, 3),
)(YOLOC1)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOVoVCSP',
    k=(3, 5),
)(YOLOVoVCSP)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOBottleneckCSPF',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOBottleneckCSPF)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOBottleneckCSPL',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOBottleneckCSPL)


YOLO_BLOCK_REGISTRY.register(
    name='YOLOBottleneckCSPLG',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(YOLOBottleneckCSPLG)


YOLO_BLOCK_REGISTRY.register(
    name='BottleneckCSPA',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(BottleneckCSPA)


YOLO_BLOCK_REGISTRY.register(
    name='BottleneckCSPB',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(BottleneckCSPB)


YOLO_BLOCK_REGISTRY.register(
    name='BottleneckCSPC',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(BottleneckCSPC)


YOLO_BLOCK_REGISTRY.register(
    name='ResCSPA',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(ResCSPA)


YOLO_BLOCK_REGISTRY.register(
    name='ResCSPB',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(ResCSPB)


YOLO_BLOCK_REGISTRY.register(
    name='ResCSPC',
    e=EXPANSION_FACTOR_RANGE,
    k=(3, 5),
    n=(1, 2, 3),
    in_e=IN_EXPANSION_FACTOR_RANGE,
)(ResCSPC)
