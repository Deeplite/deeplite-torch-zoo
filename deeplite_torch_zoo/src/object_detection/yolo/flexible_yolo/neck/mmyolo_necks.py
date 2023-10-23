from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN
from mmyolo.models.necks.yolov7_pafpn import YOLOv7PAFPN


class YOLOv8PAFPNWrapper(YOLOv8PAFPN):
    def __init__(
            self,
            ch=(256, 512, 1024),
            channel_outs=(256, 512, 1024),
        ):
        super().__init__(
            in_channels=ch,
            out_channels=channel_outs,
            deepen_factor=1.0,
            widen_factor=1.0,
            act_cfg=dict(type='SiLU', inplace=True),
        )
        self.out_shape = channel_outs


class YOLOv7PAFPNWrapper(YOLOv7PAFPN):
    def __init__(
            self,
            ch=(256, 512, 1024),
            channel_outs=(256, 512, 1024),
        ):
        super().__init__(
            in_channels=ch,
            out_channels=(ch // 2 for ch in channel_outs),
            deepen_factor=1.0,
            widen_factor=1.0,
            block_cfg=dict(
                type='ELANBlock',
                middle_ratio=0.5,
                block_ratio=0.25,
                num_blocks=4,
                num_convs_in_block=1
            ),
            upsample_feats_cat_first=False,
            act_cfg=dict(type='SiLU', inplace=True),
        )
        self.out_shape = channel_outs
