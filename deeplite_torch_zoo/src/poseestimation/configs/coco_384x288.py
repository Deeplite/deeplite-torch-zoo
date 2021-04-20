evaluation = dict(interval=10, metric="mAP", key_indicator="AP")

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
)


data_cfg = dict(
    image_size=[288, 384],
    heatmap_size=[72, 96],
    num_output_channels=channel_cfg["num_output_channels"],
    num_joints=channel_cfg["dataset_joints"],
    dataset_channel=channel_cfg["dataset_channel"],
    inference_channel=channel_cfg["inference_channel"],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    bbox_thr=1.0,
    use_gt_bbox=False,
    image_thr=0.0,
    bbox_file="/neutrino/datasets/coco2017/person_detection_results/"
    "COCO_val2017_detections_AP_H_56_person.json",
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="TopDownRandomFlip", flip_prob=0.5),
    dict(type="TopDownHalfBodyTransform", num_joints_half_body=8, prob_half_body=0.3),
    dict(type="TopDownGetRandomScaleRotation", rot_factor=40, scale_factor=0.5),
    dict(type="TopDownAffine"),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type="TopDownGenerateTarget", sigma=3),
    dict(
        type="Collect",
        keys=["img", "target", "target_weight"],
        meta_keys=[
            "image_file",
            "joints_3d",
            "joints_3d_visible",
            "center",
            "scale",
            "rotation",
            "bbox_score",
            "flip_pairs",
        ],
    ),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    #dict(type="TopDownAffine"),
    dict(type="ToTensor"),
    #dict(type="NormalizeTensor", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=[
            "image_file",
            "center",
            "scale",
            "rotation",
            "bbox_score",
            "flip_pairs",
        ],
    ),
]

test_pipeline = val_pipeline

data_root = "/neutrino/datasets/coco2017"
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type="TopDownCocoDataset",
        ann_file=f"{data_root}/annotations/person_keypoints_train2017.json",
        img_prefix=f"{data_root}/train2017/",
        data_cfg=data_cfg,
        pipeline=train_pipeline,
    ),
    val=dict(
        type="TopDownCocoDataset",
        ann_file=f"{data_root}/annotations/person_keypoints_val2017.json",
        img_prefix=f"{data_root}/val2017/",
        data_cfg=data_cfg,
        pipeline=val_pipeline,
    ),
    test=dict(
        type="TopDownCocoDataset",
        ann_file=f"{data_root}/annotations/person_keypoints_val2017.json",
        img_prefix=f"{data_root}/val2017/",
        data_cfg=data_cfg,
        pipeline=val_pipeline,
    ),
)
