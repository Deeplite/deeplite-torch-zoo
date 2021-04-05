MISSING_IDS = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
DATA = {
    "CLASSES": [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ],
    "NUM": 80,
}

MODEL = {
    "ANCHORS": [
        [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
        [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
        [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)],
    ],  # Anchors for big obj
    "STRIDES": [8, 16, 32],
    "ANCHORS_PER_SCLAE": 3,
}

# train
TRAIN = {
    "TRAIN_IMG_SIZE": 448,
    "AUGMENT": True,
    "BATCH_SIZE": 8,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 50,
    "NUMBER_WORKERS": 4,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 0,  # 2  # or None
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


# EPSILON is used to avoid computation instability like NaN or Inf.
EPSILON = 1e-9
