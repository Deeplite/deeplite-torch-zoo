# Change classes if you want to train or evaluate for subset of classes
# For instance if you want to train/eval for 'bird' and 'person' then "CLASSES": ['bird', 'person']
DATA = {
    "ALLCLASSES": [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ],
    "CLASSES": [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ],
    "CLASSES_1": ["person"],
    "CLASSES_2": ["car", "person"],
    "CLASSES_3": ["person", "dog", "car"],
    "NUM": 20,
}

MODEL = {
    "strides": [8, 16, 32],
}

TRAIN = {
    # optimization:
    "lr0": 0.0032,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    "lrf": 0.12,  # final OneCycleLR learning rate (lr0 * lrf)
    "momentum": 0.843,  # SGD momentum/Adam beta1
    "weight_decay": 0.00036,  # optimizer weight decay 5e-4
    "warmup_epochs": 2,  # warmup epochs (fractions ok)
    "warmup_momentum": 0.5,  # warmup initial momentum
    "warmup_bias_lr": 0.05,  # warmup initial bias lr
    # loss:
    "giou": 0.0296,  # box loss gain
    "cls": 0.243,  # cls loss gain
    "cls_pw": 0.631,  # cls BCELoss positive_weight
    "obj": 0.301,  # obj loss gain (scale with pixels)
    "obj_pw": 0.911,  # obj BCELoss positive_weight
    "iou_t": 0.2,  # IoU training threshold
    "anchor_t": 2.91,  # anchor-multiple threshold
    "fl_gamma": 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
    # augmentations:
    "hsv_h": 0.0138,  # image HSV-Hue augmentation (fraction)
    "hsv_s": 0.664,  # image HSV-Saturation augmentation (fraction)
    "hsv_v": 0.464,  # image HSV-Value augmentation (fraction)
    "degrees": 0.373,  # image rotation (+/- deg)
    "translate": 0.245,  # image translation (+/- fraction)
    "scale": 0.898,  # image scale (+/- gain)
    "shear": 0.602,  # image shear (+/- deg)
    "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
    "flipud": 0.00856,  # image flip up-down (probability)
    "fliplr": 0.5,  # image flip left-right (probability)
    "mosaic": 0.0,  # image mosaic (probability)
    "mixup": 0.0,  # image mixup (probability)
}
