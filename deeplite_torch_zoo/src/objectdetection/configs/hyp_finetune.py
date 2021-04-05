MODEL = {
    "ANCHORS": [
        [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
        [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
        [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)],
    ],  # Anchors for big obj
    "STRIDES": [8, 16, 32],
    "ANCHORS_PER_SCLAE": 3,
}

# test
TEST = {
    "TEST_IMG_SIZE": 544,
    "BATCH_SIZE": 1,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.01,
    "NMS_THRESH": 0.5,
    "MULTI_SCALE_TEST": False,
    "FLIP_TEST": False,
}


TRAIN = {
    "TRAIN_IMG_SIZE": 448,
    "AUGMENT": True,
    "BATCH_SIZE": 8,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 51,
    "NUMBER_WORKERS": 4,
    "LR_INIT": 0.0032,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    "LR_END": 0.12,  # final OneCycleLR learning rate (lr0 * lrf)
    "MOMENTUM": 0.843,  # SGD momentum/Adam beta1
    "WEIGHT_DECAY": 0.00036,  # optimizer weight decay 5e-4
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
    "WARMUP_EPOCHS": 2,
}
