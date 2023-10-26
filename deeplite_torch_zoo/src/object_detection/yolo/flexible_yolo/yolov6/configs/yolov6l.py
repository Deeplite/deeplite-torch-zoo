# https://github.com/meituan/YOLOv6/

# YOLOv6l model
model = dict(
    type='YOLOv6l',
    pretrained=None,
    depth_multiple=1.0,
    width_multiple=1.0,
    backbone=dict(
        type='CSPBepBackbone',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        csp_e=float(1) / 2,
        fuse_P2=True,
    ),
    neck=dict(
        type='CSPRepBiFPANNeck',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        csp_e=float(1) / 2,
    ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=3,
        anchors_init=[
            [10, 13, 19, 19, 33, 23],
            [30, 61, 59, 59, 59, 119],
            [116, 90, 185, 185, 373, 326],
        ],
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='giou',
        use_dfl=True,
        reg_max=16,  # if use_dfl is False, please set reg_max to 0
        distill_weight={
            'class': 2.0,
            'dfl': 1.0,
        },
    ),
)

training_mode = "conv_silu"
# use normal conv to speed up training and further improve accuracy.
