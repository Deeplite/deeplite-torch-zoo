MODEL = {
    "strides": [8, 16, 32],
}

TEST = {
    "conf_thresh": 0.01,
    "nms_thresh": 0.5,
}

TRAIN = {
    "train_img_size": 448,
    "multi_scale_train": False,
    # optimization:
    "epochs": 50,
    "lr0": 0.0032,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    "lrf": 0.12,  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": 0.843,  # SGD momentum/Adam beta1
    "weight_decay": 0.00036,  # optimizer weight decay 5e-4
    "warmup_epochs": 2,  # warmup epochs (fractions ok)
    "warmup_momentum": 0.8,  # warmup initial momentum
    "warmup_bias_lr": 0.1,  # warmup initial bias lr
    # loss:
    "giou": 0.0296,  # box loss gain
    "cls": 0.243,  # cls loss gain
    "cls_pw": 0.911,  # cls BCELoss positive_weight
    "obj": 0.301,  # obj loss gain (scale with pixels)
    "obj_pw": 1.0,  # obj BCELoss positive_weight
    "iou_t": 0.2,  # IoU training threshold
    "anchor_t": 2.91,  # anchor-multiple threshold
    # anchors: 0  # anchors per output grid (0 to ignore)
    "fl_gamma": 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
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
