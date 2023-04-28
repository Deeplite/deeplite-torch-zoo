from deeplite_torch_zoo.src.objectdetection.flexible_yolo.yolov6.layers.common import *
from deeplite_torch_zoo.src.objectdetection.flexible_yolo.yolov6.models.efficientrep import *
from deeplite_torch_zoo.src.objectdetection.flexible_yolo.yolov6.models.reppan import *
from deeplite_torch_zoo.src.objectdetection.yolov5.utils.general import \
    make_divisible


def build_network(config, channels=3):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

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
            cspsppf=cspsppf
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
        )

    chx = [6, 8, 10] if config.head.num_layers == 3 else [8, 9, 10, 11]
    head_channels = [channels_list[ch_idx] for ch_idx in chx]
    return backbone, neck, head_channels
