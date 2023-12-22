# YOLOv6-lite-s model
model = dict(
    type='YOLOv6-lite-s',
    pretrained=None,
    width_multiple=0.7,
    backbone=dict(
        type='Lite_EffiBackbone',
        num_repeats=[1, 3, 7, 3],
        out_channels=[24, 32, 64, 128, 256],
        scale_size=0.5,
        ),
    neck=dict(
        type='Lite_EffiNeck',
        in_channels=[256, 128, 64],
        unified_channels=96
        ),
    head=dict(
        type='Lite_EffideHead',
        in_channels=[96, 96, 96, 96],
        num_layers=4,
        anchors=1,
        strides=[8, 16, 32, 64],
        atss_warmup_epoch=4,
        iou_type='siou',
        use_dfl=False,
        reg_max=0 #if use_dfl is False, please set reg_max to 0
    )
)
