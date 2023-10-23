# https://github.com/meituan/YOLOv6/

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6.layers.common import *
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6.models.efficientrep import *
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6.models.reppan import *
from deeplite_torch_zoo.utils import make_divisible


def build_network(config, channels=3):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    num_repeat = [
        (max(round(i * depth_mul), 1) if i > 1 else i)
        for i in (num_repeat_backbone + num_repeat_neck)
    ]
    channels_list = [
        make_divisible(i * width_mul, 8)
        for i in (channels_list_backbone + channels_list_neck)
    ]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    if 'CSP' in config.model.backbone.type:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf,
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e,
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf,
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
        )

    chx = [6, 8, 10] if config.model.head.num_layers == 3 else [8, 9, 10, 11]
    head_channels = [channels_list[ch_idx] for ch_idx in chx]
    return backbone, neck, head_channels


def build_network_lite(config, channels=3):
    width_mul = config.model.width_multiple

    num_repeat_backbone = config.model.backbone.num_repeats
    out_channels_backbone = config.model.backbone.out_channels
    scale_size_backbone = config.model.backbone.scale_size
    in_channels_neck = config.model.neck.in_channels
    unified_channels_neck = config.model.neck.unified_channels
    in_channels_head = config.model.head.in_channels

    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    out_channels_backbone = [make_divisible(i * width_mul, divisor=16)
                            for i in out_channels_backbone]
    mid_channels_backbone = [make_divisible(int(i * scale_size_backbone), divisor=8)
                            for i in out_channels_backbone]
    in_channels_neck = [make_divisible(i * width_mul, divisor=16)
                       for i in in_channels_neck]

    backbone = BACKBONE(channels,
                        mid_channels_backbone,
                        out_channels_backbone,
                        num_repeat=num_repeat_backbone)
    neck = NECK(in_channels_neck, unified_channels_neck)

    return backbone, neck, in_channels_head
