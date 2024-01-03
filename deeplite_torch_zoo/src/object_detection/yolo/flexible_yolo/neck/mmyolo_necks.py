from mmyolo.models.necks.yolov7_pafpn import YOLOv7PAFPN
from mmyolo.models.necks.ppyoloe_csppan import PPYOLOECSPPAFPN

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.neck.neck_utils import YOLO_SCALING_GAINS, NECK_ACT_TYPE_MAP


class YOLOv7PAFPNWrapper(YOLOv7PAFPN):
    def __init__(
            self,
            ch=(256, 512, 1024),
            channel_outs=(256, 512, 1024),
            version='l',
            width_factor=None,
            depth_factor=None,
            act='silu',
            upsample_feats_cat_first=False,
            elan_middle_ratio=0.5,
            elan_block_ratio=0.25,
            elan_num_blocks=4,
            elan_num_convs_in_block=1,
        ):
        if width_factor is None:
            width_factor = YOLO_SCALING_GAINS[version.lower()]['gw']
        if depth_factor is None:
            depth_factor = YOLO_SCALING_GAINS[version.lower()]['gd']
        super().__init__(
            in_channels=ch,
            out_channels=(channels // 2 for channels in channel_outs),
            deepen_factor=depth_factor,
            widen_factor=width_factor,
            block_cfg=dict(
                type='ELANBlock',
                middle_ratio=elan_middle_ratio,
                block_ratio=elan_block_ratio,
                num_blocks=elan_num_blocks,
                num_convs_in_block=elan_num_convs_in_block,
            ),
            upsample_feats_cat_first=upsample_feats_cat_first,
            act_cfg=NECK_ACT_TYPE_MAP[act],
        )
        self.out_shape = channel_outs


class PPYOLOECSPPAFPNWrapper(PPYOLOECSPPAFPN):
    def __init__(
            self,
            ch=(256, 512, 1024),
            channel_outs=(256, 512, 1024),
            version='l',
            width_factor=None,
            depth_factor=None,
            act='silu',
            num_csplayer=1,
            num_blocks_per_layer=3,
            drop_block_cfg=None,
            use_spp=True,
        ):
        if width_factor is None:
            width_factor = YOLO_SCALING_GAINS[version.lower()]['gw']
        if depth_factor is None:
            depth_factor = YOLO_SCALING_GAINS[version.lower()]['gd']
        super().__init__(
            in_channels=ch,
            out_channels=channel_outs,
            deepen_factor=depth_factor,
            widen_factor=width_factor,
            act_cfg=NECK_ACT_TYPE_MAP[act],
            num_csplayer=num_csplayer,
            num_blocks_per_layer=num_blocks_per_layer,
            block_cfg=dict(
                type='PPYOLOEBasicBlock', shortcut=False, use_alpha=False),
            norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
            drop_block_cfg=drop_block_cfg,
            use_spp=use_spp,
        )
        self.out_shape = channel_outs
