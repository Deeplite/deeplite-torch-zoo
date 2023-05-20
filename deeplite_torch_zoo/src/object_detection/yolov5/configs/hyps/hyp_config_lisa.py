DATA = {
    "CLASSES": [
        "speedLimit15",
        "speedLimit25",
        "speedLimit30",
        "speedLimit35",
        "speedLimit40",
        "speedLimit45",
        "speedLimit50",
        "speedLimit55",
        "speedLimit65",
        "truckSpeedLimit55",
        "speedLimitUrdbl",
    ],
    "NUM": 11,
}

MODEL = {
    "strides": [8, 16, 32],
}

TRAIN = {
    # optimization:
    "epochs": 50,
    "lr0": 1e-4,  # 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    "lrf": 1e-2,  # 0.2,  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": 0.937,  # SGD momentum/Adam beta1
    "weight_decay": 0.0005,  # optimizer weight decay 5e-4
    "warmup_epochs": 2,  # warmup epochs (fractions ok)
    "warmup_momentum": 0.8,  # warmup initial momentum
    "warmup_bias_lr": 0.1,  # warmup initial bias lr
    # loss:
    "giou": 0.05,  # box loss gain
    "cls": 0.5,  # cls loss gain
    "cls_pw": 1.0,  # cls BCELoss positive_weight
    "obj": 1.0,  # obj loss gain (scale with pixels)
    "obj_pw": 1.0,  # obj BCELoss positive_weight
    "iou_t": 0.20,  # IoU training threshold
    "anchor_t": 4.0,  # anchor-multiple threshold
    # anchors: 0  # anchors per output grid (0 to ignore)
    "fl_gamma": 1.5,  # focal loss gamma (efficientDet default gamma=1.5)
    "giou_loss_ratio": 1.0,
    # augmentations:
    "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
    "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
    "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
    "degrees": 0.0,  # image rotation (+/- deg)
    "translate": 0.1,  # image translation (+/- fraction)
    "scale": 0.5,  # image scale (+/- gain)
    "shear": 0.0,  # image shear (+/- deg)
    "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
    "flipud": 0.0,  # image flip up-down (probability)
    "fliplr": 0.5,  # image flip left-right (probability)
    "mosaic": 0.0,  # image mosaic (probability)
    "mixup": 0.5,  # image mixup (probability)
    "copy_paste": 0.0,  # segment copy-paste (probability)
}
