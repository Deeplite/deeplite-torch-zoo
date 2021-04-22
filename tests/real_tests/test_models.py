import unittest

import pytest
import torch

from pycocotools.coco import COCO

from deeplite_torch_zoo.wrappers.wrapper import get_model_by_name, get_data_splits_by_name

from deeplite_torch_zoo.wrappers.eval import (yolo_eval_func, seg_eval_func, rcnn_eval_coco,
    vgg16_ssd_eval_func, mb1_ssd_eval_func, mb2_ssd_eval_func, mb2_ssd_lite_eval_func,
    keypoint_rcnn_eval_coco)


class TestModels(unittest.TestCase):

    @pytest.mark.test_keypointrcnn_resnet50_fpn_coco
    def test_keypointrcnn_resnet50_fpn_coco(self):
        model = get_model_by_name(
            model_name="keypointrcnn_resnet50_fpn",
            dataset_name="coco_80",
            pretrained=True,
            progress=False,
        )
        test_loader = get_data_splits_by_name(
            dataset_name="coco",
            model_name="keypointrcnn_resnet50_fpn",
        )["test"]
        res = keypoint_rcnn_eval_coco(
            model, data_loader=test_loader, _set="coco_384x288"
        )
        print(res)
        mAP = res["AP"]
        self.assertEqual(abs(mAP - 0.639) < 0.001, True)

    @pytest.mark.test_vgg16_ssd_voc
    def test_vgg16_ssd_voc(self):
        model = get_model_by_name(
            model_name="vgg16_ssd",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        test_loader = get_data_splits_by_name(
            data_root="/neutrino/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="vgg16_ssd",
            batch_size=32,
        )["test"]
        APs = vgg16_ssd_eval_func(model, test_loader)
        self.assertEqual(abs(APs["mAP"] - 0.7731) < 0.001, True)

    @pytest.mark.test_mb1_ssd_voc
    def test_mb1_ssd_voc(self):
        model = get_model_by_name(
            model_name="mb1_ssd",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        test_loader = get_data_splits_by_name(
            data_root="/neutrino/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="mb1_ssd",
            batch_size=32,
        )["test"]
        APs = mb1_ssd_eval_func(model, test_loader)
        self.assertEqual(abs(APs["mAP"] - 0.6755) < 0.001, True)

    @pytest.mark.test_mb2_ssd_lite_voc_20
    def test_mb2_ssd_lite_voc_20(self):
        model = get_model_by_name(
            model_name="mb2_ssd_lite",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        test_loader = get_data_splits_by_name(
            data_root="/neutrino/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="mb2_ssd_lite",
            batch_size=32,
        )["test"]
        APs = mb2_ssd_lite_eval_func(
            model, test_loader
        )
        self.assertEqual(abs(APs["mAP"] - 0.687) < 0.001, True)

    @pytest.mark.test_mb2_ssd_voc_20
    def test_mb2_ssd_voc_20(self):
        model = get_model_by_name(
            model_name="mb2_ssd",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        test_loader = get_data_splits_by_name(
            data_root="/neutrino/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="mb2_ssd_lite",
            batch_size=32,
        )["test"]
        APs = mb2_ssd_lite_eval_func(
            model, test_loader
        )
        self.assertEqual(abs(APs["mAP"] - 0.443) < 0.001, True)

    @pytest.mark.test_mb2_ssd_coco_80
    def test_mb2_ssd_coco_80(self):
        model = get_model_by_name(
            model_name="mb2_ssd",
            dataset_name="coco_80",
            pretrained=True,
            progress=False,
        )
        from deeplite_torch_zoo.src.objectdetection.configs.coco_config import DATA, MISSING_IDS
        test_loader = get_data_splits_by_name(
            data_root="/neutrino/datasets/coco2017/",
            dataset_name="coco",
            model_name="mb2_ssd",
            batch_size=32,
            missing_ids=MISSING_IDS,
            classes=DATA["CLASSES"],
        )["test"]
        cocoGt = COCO("/neutrino/datasets/coco2017/annotations/instances_val2017.json")
        APs = mb2_ssd_eval_func(
            model, test_loader, gt=cocoGt, _set="coco",
        )
        self.assertEqual(abs(APs["mAP"] - 0.303) < 0.001, True)

    @pytest.mark.test_mb2_ssd_coco_6
    def test_mb2_ssd_coco_6(self):
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
        APs = mb2_ssd_eval_func(
            model, test_loader, gt=cocoGt, _set="coco",
        )
        self.assertEqual(abs(APs["mAP"] - 0.227) < 0.001, True)

    @pytest.mark.test_mb2_ssd_lite_voc_1
    def test_mb2_ssd_lite_voc_1(self):
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
        APs = mb2_ssd_lite_eval_func(
            model, test_loader
        )
        self.assertEqual(abs(APs["mAP"] - 0.664) < 0.001, True)

    @pytest.mark.test_mb2_ssd_lite_voc_2
    def test_mb2_ssd_lite_voc_2(self):
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
        APs = mb2_ssd_lite_eval_func(
            model, test_loader
        )
        self.assertEqual(abs(APs["mAP"] - 0.716) < 0.001, True)

    @pytest.mark.test_yolov3_voc_20
    def test_yolov3_voc_20(self):
        model = get_model_by_name(
            model_name="yolo3",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov3"
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.829) < 0.001, True)

    @pytest.mark.test_yolov3_voc_1
    def test_yolov3_voc_1(self):
        model = get_model_by_name(
            model_name="yolo3",
            dataset_name="voc_1",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model,
            "/neutrino/datasets/VOCdevkit/VOC2007/",
            num_classes=1,
            _set="voc",
            net="yolov3_1cls",
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.888) < 0.001, True)

    @pytest.mark.test_yolov3_voc_2
    def test_yolov3_voc_2(self):
        model = get_model_by_name(
            model_name="yolo3",
            dataset_name="voc_2",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model,
            "/neutrino/datasets/VOCdevkit/VOC2007/",
            num_classes=2,
            _set="voc",
            net="yolov3_2cls",
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.911) < 0.001, True)

    @pytest.mark.skip(
        reason="we don't know which 6 classes are used to train that model"
    )
    @pytest.mark.test_yolov3_voc_6
    def test_yolov3_voc_6(self):
        model = get_model_by_name(
            model_name="yolo3",
            dataset_name="voc_6",
            pretrained=True,
            progress=False,
        )

        model = yolo3_voc_6(pretrained=True, progress=False)
        APs = yolo_eval_func(
            model,
            "/neutrino/datasets/VOCdevkit/VOC2007/",
            _set="voc",
            net="yolov3_6cls",
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.839) < 0.001, True)

    @pytest.mark.test_yolov4s_voc
    def test_yolov4s_voc(self):
        model = get_model_by_name(
            model_name="yolo4s",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4s"
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.857) < 0.001, True)

    @pytest.mark.test_yolov4m_voc
    def test_yolov4m_voc(self):
        model = get_model_by_name(
            model_name="yolo4m",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4m"
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.882) < 0.001, True)

    @pytest.mark.test_yolov4l_voc
    def test_yolov4l_voc(self):
        model = get_model_by_name(
            model_name="yolo4l",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4l"
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.882) < 0.001, True)

    @pytest.mark.test_yolov4x_voc
    def test_yolov4x_voc(self):
        model = get_model_by_name(
            model_name="yolo4x",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4x"
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.893) < 0.001, True)

    @pytest.mark.skip(reason="not compatible with pytorch 1.4")
    def test_yolov5s_voc(self):
        model = get_model_by_name(
            model_name="yolo5s",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model, "/neutrino/datasets//VOCdevkit/VOC2007/", _set="voc"
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.817) < 0.001, True)

    @pytest.mark.skip(reason="not compatible with pytorch 1.4")
    def test_yolov5m_voc(self):
        model = get_model_by_name(
            model_name="yolo5m",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model, "/neutrino/datasets//VOC/VOCdevkit/VOC2007/", _set="voc"
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.882) < 0.001, True)

    @pytest.mark.skip(reason="not compatible with pytorch 1.4")
    def test_yolov5l_voc(self):
        model = get_model_by_name(
            model_name="yolo5l",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc")
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.900) < 0.001, True)

    @pytest.mark.skip(reason="not compatible with pytorch 1.4")
    def test_yolov5x_voc(self):
        model = get_model_by_name(
            model_name="yolo5x",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model, "/neutrino/datasetss/VOCdevkit/VOC2007/", _set="voc"
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.905) < 0.001, True)

    @pytest.mark.skip(reason="not compatible with pytorch 1.4")
    def test_yolov5s_coco(self):
        model = get_model_by_name(
            model_name="yolo5s",
            dataset_name="coco_80",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(model, "/neutrino/datasets/coco", _set="coco")
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.905) < 0.001, True)

    @pytest.mark.skip(reason="not compatible with pytorch 1.4")
    def test_yolov5m_coco(self):
        model = get_model_by_name(
            model_name="yolo5m",
            dataset_name="coco_80",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(model, "/neutrino/datasets/coco", _set="coco")
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.905) < 0.001, True)

    @pytest.mark.skip(reason="not compatible with pytorch 1.4")
    def test_yolov5l_coco(self):
        model = get_model_by_name(
            model_name="yolo5l",
            dataset_name="coco_80",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(model, "/neutrino/datasets/coco", _set="coco")
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.905) < 0.001, True)

    @pytest.mark.skip(reason="not compatible with pytorch 1.4")
    def test_yolov5x_coco(self):
        model = get_model_by_name(
            model_name="yolo5x",
            dataset_name="coco_80",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(model, "/neutrino/datasets/coco", _set="coco")
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.905) < 0.001, True)

    @pytest.mark.test_ssd300_resnet18_voc
    def test_ssd300_resnet18_voc(self):
        model = get_model_by_name(
            model_name="ssd300_resnet18",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model,
            "/neutrino/datasets/VOCdevkit/VOC2007",
            _set="voc",
            img_size=300,
            net="ssd300_resnet18",
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.58) < 0.001, True)

    @pytest.mark.test_ssd300_resnet34_voc
    def test_ssd300_resnet34_voc(self):
        model = get_model_by_name(
            model_name="ssd300_resnet34",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model,
            "/neutrino/datasets/VOCdevkit/VOC2007",
            _set="voc",
            img_size=300,
            net="ssd300_resnet34",
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.654) < 0.001, True)

    @pytest.mark.test_ssd300_resnet50_voc
    def test_ssd300_resnet50_voc(self):
        model = get_model_by_name(
            model_name="ssd300_resnet50",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model,
            "/neutrino/datasets/VOCdevkit/VOC2007",
            _set="voc",
            img_size=300,
            net="ssd300_resnet50",
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.659) < 0.001, True)

    @pytest.mark.test_ssd300_vgg16_voc
    def test_ssd300_vgg16_voc(self):
        model = get_model_by_name(
            model_name="ssd300_vgg16",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(
            model,
            "/neutrino/datasets/VOCdevkit/VOC2007",
            _set="voc",
            img_size=300,
            net="ssd300_vgg16",
        )
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.641) < 0.001, True)

    @pytest.mark.test_yolov4m_lisa
    def test_yolov4m_lisa(self):
        model = get_model_by_name(
            model_name="yolo4m",
            dataset_name="lisa_11",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(model, "/neutrino/datasets/lisa", _set="lisa")
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.968) < 0.001, True)

    @pytest.mark.test_yolov3_lisa
    def test_yolov3_lisa(self):
        model = get_model_by_name(
            model_name="yolo3",
            dataset_name="lisa_11",
            pretrained=True,
            progress=False,
        )
        APs = yolo_eval_func(model, "/neutrino/datasets/lisa", _set="lisa")
        print(APs)
        self.assertEqual(abs(APs["mAP"] - 0.836) < 0.001, True)

    @pytest.mark.test_fasterrcnn_resnet50_fpn_coco
    def test_fasterrcnn_resnet50_fpn_coco(self):
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
        APs = rcnn_eval_coco(
            model, test_loader, gt=cocoGt
        )
        self.assertEqual(abs(APs["mAP"] - 0.369) < 0.001, True)

    @pytest.mark.test_maskrcnn_resnet50_fpn_coco
    def test_maskrcnn_resnet50_fpn_coco(self):
        model = get_model_by_name(
            model_name="maskrcnn_resnet50_fpn",
            dataset_name="coco_80",
            pretrained=True,
            progress=False,
        )
        test_loader = get_data_splits_by_name(
            data_root="/neutrino/datasets/coco2017/",
            dataset_name="coco",
            model_name="maskrcnn_resnet50_fpn",
            batch_size=32,
        )["test"]
        cocoGt = COCO("/neutrino/datasets/coco2017/annotations/instances_val2017.json")
        APs = rcnn_eval_coco(
            model, test_loader, gt=cocoGt
        )
        self.assertEqual(abs(APs["mAP"] - 0.3789) < 0.001, True)

    @pytest.mark.test_unet_carvana
    def test_unet_carvana(self):
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
        acc = seg_eval_func(model, test_loader, net="unet")
        dc = acc["dice_coeff"]
        print(dc)
        self.assertEqual(abs(dc - 0.983) < 0.001, True)

    @pytest.mark.test_unet_scse_resnet18_voc_20
    def test_unet_scse_resnet18_voc_20(self):
        model = get_model_by_name(
            model_name="unet_scse_resnet18",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        test_loader = get_data_splits_by_name(
            data_root="/neutrino/datasets",
            dataset_name="voc",
            model_name="unet",
            num_workers=1,
        )["test"]
        acc = seg_eval_func(model, test_loader, net="unet_scse_resnet18")
        miou = acc["miou"]
        print(miou)
        self.assertEqual(abs(miou - 0.582) < 0.001, True)

    @pytest.mark.test_unet_scse_resnet18_voc_1
    def test_unet_scse_resnet18_voc_1(self):
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
        acc = seg_eval_func(model, test_loader, net="unet_scse_resnet18")
        miou = acc["miou"]
        print(miou)
        self.assertEqual(abs(miou - 0.673) < 0.001, True)

    @pytest.mark.test_unet_scse_resnet18_voc_2
    def test_unet_scse_resnet18_voc_2(self):
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
        acc = seg_eval_func(model, test_loader, net="unet_scse_resnet18")
        miou = acc["miou"]
        print(miou)
        self.assertEqual(abs(miou - 0.679) < 0.001, True)

    @pytest.mark.test_unet_scse_resnet18_carvana
    def test_unet_scse_resnet18_carvana(self):
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
        acc = seg_eval_func(model, test_loader, net="unet_scse_resnet18")
        miou = acc["miou"]
        print(miou)
        self.assertEqual(abs(miou - 0.989) < 0.001, True)

    @pytest.mark.test_fcn32_voc_20
    def test_fcn32_voc_20(self):
        model = get_model_by_name(
            model_name="fcn32",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
        )
        test_loader = get_data_splits_by_name(
            data_root="/neutrino/datasets",
            dataset_name="voc",
            model_name="fcn32",
            num_workers=1,
            backbone="vgg",
        )["test"]
        acc = seg_eval_func(model, test_loader, net="fcn32")
        miou = acc["miou"]
        print(miou)
        self.assertEqual(abs(miou - 0.713) < 0.001, True)

    @pytest.mark.test_deeplab_mobilenet_voc_20
    def test_deeplab_mobilenet_voc_20(self):
        model = get_model_by_name(
            model_name="deeplab_mobilenet",
            dataset_name="voc_20",
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
        acc = seg_eval_func(model, test_loader, net="deeplab")
        miou = acc["miou"]
        print(miou)
        self.assertEqual(abs(miou - 0.571) < 0.001, True)


if __name__ == "__main__":
    unittest.main()
