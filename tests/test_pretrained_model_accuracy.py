import pytest
from pycocotools.coco import COCO

from deeplite_torch_zoo import (get_data_splits_by_name, get_eval_function,
                                get_model_by_name)


@pytest.mark.slow
def test_vgg19_tinyimagenet():
    model = get_model_by_name(
        model_name="vgg19",
        dataset_name="tinyimagenet",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/TinyImageNet/",
        dataset_name="tinyimagenet",
        batch_size=128,
        num_workers=0,
    )["val"]
    eval_fn = get_eval_function("vgg19", "tinyimagenet")
    ACC = eval_fn(model, test_loader)
    assert abs(ACC["acc"] - 0.728) < 0.001


@pytest.mark.slow
def test_mobilenet_v2_tinyimagenet():
    model = get_model_by_name(
        model_name="mobilenet_v2",
        dataset_name="tinyimagenet",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/TinyImageNet/",
        dataset_name="tinyimagenet",
        batch_size=128,
        num_workers=0,
    )["val"]
    eval_fn = get_eval_function("mobilenet_v2", "tinyimagenet")
    ACC = eval_fn(model, test_loader)

    assert abs(ACC["acc"] - 0.680) < 0.001


@pytest.mark.slow
def test_resnet18_tinyimagenet():
    model = get_model_by_name(
        model_name="resnet18",
        dataset_name="tinyimagenet",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/TinyImageNet/",
        dataset_name="tinyimagenet",
        batch_size=128,
        num_workers=0,
    )["val"]
    eval_fn = get_eval_function("resnet18", "tinyimagenet")
    ACC = eval_fn(model, test_loader)

    assert abs(ACC["acc"] - 0.663) < 0.001


@pytest.mark.slow
def test_resnet34_tinyimagenet():
    model = get_model_by_name(
        model_name="resnet34",
        dataset_name="tinyimagenet",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/TinyImageNet/",
        dataset_name="tinyimagenet",
        batch_size=128,
        num_workers=0,
    )["val"]
    eval_fn = get_eval_function("resnet34", "tinyimagenet")
    ACC = eval_fn(model, test_loader)

    assert abs(ACC["acc"] - 0.686) < 0.001


@pytest.mark.slow
def test_resnet50_tinyimagenet():
    model = get_model_by_name(
        model_name="resnet50",
        dataset_name="tinyimagenet",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/TinyImageNet/",
        dataset_name="tinyimagenet",
        batch_size=128,
        num_workers=0,
    )["val"]
    eval_fn = get_eval_function("resnet50", "tinyimagenet")
    ACC = eval_fn(model, test_loader)

    assert abs(ACC["acc"] - 0.730) < 0.001


@pytest.mark.slow
def test_mb3_large_vww():
    model = get_model_by_name(
        model_name="mobilenetv3_large",
        dataset_name="vww",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/vww",
        dataset_name="vww",
        batch_size=128,
    )["test"]
    eval_fn = get_eval_function("mobilenetv3_large", "vww")
    ACC = eval_fn(model, test_loader)
    assert abs(ACC["acc"] - 0.891) < 0.001


@pytest.mark.slow
def test_mb3_small_vww():
    model = get_model_by_name(
        model_name="mobilenetv3_small",
        dataset_name="vww",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/vww",
        dataset_name="vww",
        batch_size=128,
    )["test"]
    eval_fn = get_eval_function("mobilenetv3_small", "vww")
    ACC = eval_fn(model, test_loader)
    assert abs(ACC["acc"] - 0.892) < 0.001


@pytest.mark.slow
def test_resnet18_ssd_voc():
    model = get_model_by_name(
        model_name="resnet18_ssd",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/VOCdevkit",
        dataset_name="voc",
        model_name="resnet18_ssd",
        batch_size=32,
    )["test"]
    eval_fn = get_eval_function("resnet18_ssd", "voc")
    APs = eval_fn(model, test_loader)
    assert abs(APs["mAP"] - 0.728) < 0.001


@pytest.mark.slow
def test_resnet34_ssd_voc():
    model = get_model_by_name(
        model_name="resnet34_ssd",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/VOCdevkit",
        dataset_name="voc",
        model_name="resnet34_ssd",
        batch_size=32,
    )["test"]
    eval_fn = get_eval_function("resnet34_ssd", "voc")
    APs = eval_fn(model, test_loader)
    assert abs(APs["mAP"] - 0.760) < 0.001

@pytest.mark.slow
def test_resnet50_ssd_voc():
    model = get_model_by_name(
        model_name="resnet50_ssd",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/VOCdevkit",
        dataset_name="voc",
        model_name="resnet50_ssd",
        batch_size=32,
    )["test"]
    eval_fn = get_eval_function("resnet50_ssd", "voc")
    APs = eval_fn(model, test_loader)
    assert abs(APs["mAP"] - 0.766) < 0.001


@pytest.mark.slow
def test_vgg16_ssd_voc():
    model = get_model_by_name(
        model_name="vgg16_ssd",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/VOCdevkit",
        dataset_name="voc",
        model_name="vgg16_ssd",
        batch_size=32,
    )["test"]
    eval_fn = get_eval_function("vgg16_ssd", "voc")
    APs = eval_fn(model, test_loader)
    assert abs(APs["mAP"] - 0.7731) < 0.001


@pytest.mark.slow
def test_vgg16_ssd_wider_face():
    model = get_model_by_name(
        model_name="vgg16_ssd",
        dataset_name="wider_face",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/wider_face",
        dataset_name="wider_face",
        model_name="vgg16_ssd",
        batch_size=8,
    )["test"]
    eval_fn = get_eval_function("vgg16_ssd", "wider_face")
    APs = eval_fn(model, test_loader)

    assert abs(APs["mAP"] - 0.7071) < 0.001


@pytest.mark.slow
def test_mb1_ssd_voc():
    model = get_model_by_name(
        model_name="mb1_ssd",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/VOCdevkit",
        dataset_name="voc",
        model_name="mb1_ssd",
        batch_size=32,
    )["test"]
    eval_fn = get_eval_function("mb1_ssd", "voc")
    APs = eval_fn(model, test_loader)
    assert abs(APs["mAP"] - 0.6755) < 0.001


@pytest.mark.slow
def test_mb2_ssd_lite_voc():
    model = get_model_by_name(
        model_name="mb2_ssd_lite",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/VOCdevkit",
        dataset_name="voc",
        model_name="mb2_ssd_lite",
        batch_size=32,
    )["test"]
    eval_fn = get_eval_function("mb2_ssd_lite", "voc")
    APs = eval_fn(model, test_loader)
    assert abs(APs["mAP"] - 0.687) < 0.001


@pytest.mark.slow
def test_mb2_ssd_voc():
    model = get_model_by_name(
        model_name="mb2_ssd",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/VOCdevkit",
        dataset_name="voc",
        model_name="mb2_ssd_lite",
        batch_size=32,
    )["test"]
    eval_fn = get_eval_function("mb2_ssd", "voc")
    APs = eval_fn(model, test_loader)
    assert abs(APs["mAP"] - 0.443) < 0.001


@pytest.mark.slow
def test_mb2_ssd_coco_80():
    model = get_model_by_name(
        model_name="mb2_ssd",
        dataset_name="coco_80",
        pretrained=True,
        progress=False,
    )
    from deeplite_torch_zoo.src.objectdetection.datasets.coco_config import (
        DATA, MISSING_IDS)

    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/coco2017/",
        dataset_name="coco",
        model_name="mb2_ssd",
        batch_size=32,
        missing_ids=MISSING_IDS,
        classes=DATA["CLASSES"],
    )["test"]
    cocoGt = COCO("/neutrino/datasets/coco2017/annotations/instances_val2017.json")

    eval_fn = get_eval_function("mb2_ssd", "coco_80")
    APs = eval_fn(
        model,
        test_loader,
        gt=cocoGt,
        _set="coco",
    )


    assert abs(APs["mAP"] - 0.138) < 0.001


@pytest.mark.slow
def test_mb2_ssd_coco_6():
    model = get_model_by_name(
        model_name="mb2_ssd",
        dataset_name="coco_gm_6",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/home/ehsan/data/",
        dataset_name="coco_gm",
        model_name="mb2_ssd",
        batch_size=32,
        train_ann_file="train_data_COCO.json",
        train_dir="images/train",
        val_ann_file="test_data_COCO.json",
        val_dir="images/test",
        classes=["class1", "class2", "class3", "class4", "class5", "class6"],
    )["test"]
    cocoGt = COCO("/home/ehsan/data/test_data_COCO.json")
    eval_fn = get_eval_function("mb2_ssd", "coco_gm")
    APs = eval_fn(
        model,
        test_loader,
        gt=cocoGt,
        _set="coco",
    )

    assert abs(APs["mAP"] - 0.227) < 0.001


@pytest.mark.slow
def test_mb2_ssd_lite_voc_1():
    model = get_model_by_name(
        model_name="mb2_ssd_lite",
        dataset_name="voc_1",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/VOCdevkit",
        dataset_name="voc",
        model_name="mb2_ssd_lite",
        num_classes=1,
        batch_size=32,
    )["test"]
    eval_fn = get_eval_function("mb2_ssd_lite", "voc_1")
    APs = eval_fn(model, test_loader)
    assert abs(APs["mAP"] - 0.664) < 0.001


@pytest.mark.slow
def test_mb2_ssd_lite_voc_2():
    model = get_model_by_name(
        model_name="mb2_ssd_lite",
        dataset_name="voc_2",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/VOCdevkit",
        dataset_name="voc",
        model_name="mb2_ssd_lite",
        num_classes=2,
        batch_size=32,
    )["test"]
    eval_fn = get_eval_function("mb2_ssd_lite", "voc_2")
    APs = eval_fn(model, test_loader)
    assert abs(APs["mAP"] - 0.716) < 0.001


@pytest.mark.slow
def test_yolov3_voc():
    model = get_model_by_name(
        model_name="yolo3",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    eval_fn = get_eval_function("yolo3", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov3"
    )

    assert abs(APs["mAP"] - 0.829) < 0.001


@pytest.mark.slow
def test_yolov4s_voc():
    model = get_model_by_name(
        model_name="yolo4s",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    eval_fn = get_eval_function("yolo4s", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4s"
    )

    assert abs(APs["mAP"] - 0.857) < 0.001

@pytest.mark.slow
def test_yolov4m_voc():
    model = get_model_by_name(
        model_name="yolo4m",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    eval_fn = get_eval_function("yolo4m", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4m"
    )

    assert abs(APs["mAP"] - 0.882) < 0.001


@pytest.mark.slow
def test_yolov4l_voc():
    model = get_model_by_name(
        model_name="yolo4l",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    eval_fn = get_eval_function("yolo4l", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolo4l"
    )

    assert abs(APs["mAP"] - 0.882) < 0.001


@pytest.mark.slow
def test_yolov4l_leaky_voc():
    model = get_model_by_name(
        model_name="yolo4l_leaky",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    eval_fn = get_eval_function("yolo4l_leaky", "voc")
    APs = eval_fn(
        model,
        "/neutrino/datasets/VOCdevkit/VOC2007/",
        _set="voc",
        net="yolov4l_leaky",
    )

    assert abs(APs["mAP"] - 0.890) < 0.001


@pytest.mark.slow
def test_yolov4x_voc():
    model = get_model_by_name(
        model_name="yolo4x",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    eval_fn = get_eval_function("yolo4x", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4x"
    )

    assert abs(APs["mAP"] - 0.893) < 0.001


@pytest.mark.slow
def test_yolov5_6s_voc():
    model = get_model_by_name(
        model_name="yolo5_6s",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    eval_fn = get_eval_function("yolo5_6s", "voc")
    APs = eval_fn(model, "/neutrino/datasets//VOCdevkit/VOC2007/", _set="voc")

    assert abs(APs["mAP"] - 0.821) < 0.001


@pytest.mark.slow
def test_fasterrcnn_resnet50_fpn_coco():
    model = get_model_by_name(
        model_name="fasterrcnn_resnet50_fpn",
        dataset_name="coco_80",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/coco2017/",
        dataset_name="coco",
        model_name="fasterrcnn_resnet50_fpn",
        batch_size=32,
    )["test"]
    cocoGt = COCO("/neutrino/datasets/coco2017/annotations/instances_val2017.json")
    eval_fn = get_eval_function("fasterrcnn_resnet50_fpn", "coco_80")
    APs = eval_fn(model, test_loader, gt=cocoGt)
    assert abs(APs["mAP"] - 0.369) < 0.001


@pytest.mark.slow
def test_unet_carvana():
    model = get_model_by_name(
        model_name="unet",
        dataset_name="carvana",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/carvana",
        dataset_name="carvana",
        model_name="unet",
        num_workers=1,
    )["test"]
    eval_fn = get_eval_function("unet", "carvana")
    acc = eval_fn(model, test_loader, net="unet")
    dc = acc["dice_coeff"]

    assert abs(dc - 0.983) < 0.001


@pytest.mark.slow
def test_unet_scse_resnet18_voc():
    model = get_model_by_name(
        model_name="unet_scse_resnet18",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets",
        dataset_name="voc",
        model_name="unet",
        num_workers=1,
    )["test"]
    eval_fn = get_eval_function("unet_scse_resnet18", "voc")
    acc = eval_fn(model, test_loader, net="unet_scse_resnet18")
    miou = acc["miou"]
    assert abs(miou - 0.582) < 0.001


@pytest.mark.slow
def test_unet_scse_resnet18_voc_1():
    model = get_model_by_name(
        model_name="unet_scse_resnet18",
        dataset_name="voc_1",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets",
        dataset_name="voc",
        model_name="unet",
        num_classes=2,
        num_workers=2,
    )["test"]
    eval_fn = get_eval_function("unet_scse_resnet18", "voc_1")
    acc = eval_fn(model, test_loader, net="unet_scse_resnet18")
    miou = acc["miou"]
    assert abs(miou - 0.673) < 0.001


@pytest.mark.slow
def test_unet_scse_resnet18_voc_2():
    model = get_model_by_name(
        model_name="unet_scse_resnet18",
        dataset_name="voc_2",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets",
        dataset_name="voc",
        model_name="unet",
        num_classes=3,
        num_workers=1,
    )["test"]
    eval_fn = get_eval_function("unet_scse_resnet18", "voc_2")
    acc = eval_fn(model, test_loader, net="unet_scse_resnet18")
    miou = acc["miou"]

    assert abs(miou - 0.679) < 0.001


@pytest.mark.slow
def test_unet_scse_resnet18_carvana():
    model = get_model_by_name(
        model_name="unet_scse_resnet18",
        dataset_name="carvana",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets/carvana",
        dataset_name="carvana",
        model_name="unet",
        num_workers=1,
    )["test"]
    eval_fn = get_eval_function("unet_scse_resnet18", "carvana")
    acc = eval_fn(model, test_loader, net="unet_scse_resnet18")
    miou = acc["miou"]

    assert abs(miou - 0.989) < 0.001


@pytest.mark.slow
def test_fcn32_voc():
    model = get_model_by_name(
        model_name="fcn32",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets",
        dataset_name="voc",
        model_name="fcn32",
        num_workers=1,
        batch_size=1,
        backbone="vgg",
    )["test"]
    eval_fn = get_eval_function("fcn32", "voc")
    acc = eval_fn(model, test_loader, net="fcn32")
    miou = acc["miou"]

    assert abs(miou - 0.713) < 0.001


@pytest.mark.slow
def test_deeplab_mobilenet_voc():
    model = get_model_by_name(
        model_name="deeplab_mobilenet",
        dataset_name="voc",
        pretrained=True,
        progress=False,
    )
    test_loader = get_data_splits_by_name(
        data_root="/neutrino/datasets",
        sbd_root=None,
        dataset_name="voc",
        model_name="deeplab_mobilenet",
        num_workers=2,
        backbone="vgg",
    )["test"]
    eval_fn = get_eval_function("deeplab_mobilenet", "voc")
    acc = eval_fn(model, test_loader, net="deeplab")
    miou = acc["miou"]

    assert abs(miou - 0.571) < 0.001
