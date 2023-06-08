import pytest

from deeplite_torch_zoo import (get_dataloaders, get_eval_function,
                                get_model)


@pytest.mark.slow
def test_vgg19_tinyimagenet():
    model = get_model(
        model_name="vgg19",
        dataset_name="tinyimagenet",
        pretrained=True,
    )
    test_loader = get_dataloaders(
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
    model = get_model(
        model_name="mobilenet_v2",
        dataset_name="tinyimagenet",
        pretrained=True,
    )
    test_loader = get_dataloaders(
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
    model = get_model(
        model_name="resnet18",
        dataset_name="tinyimagenet",
        pretrained=True,
    )
    test_loader = get_dataloaders(
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
    model = get_model(
        model_name="resnet34",
        dataset_name="tinyimagenet",
        pretrained=True,
    )
    test_loader = get_dataloaders(
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
    model = get_model(
        model_name="resnet50",
        dataset_name="tinyimagenet",
        pretrained=True,
    )
    test_loader = get_dataloaders(
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
    model = get_model(
        model_name="mobilenetv3_large",
        dataset_name="vww",
        pretrained=True,
    )
    test_loader = get_dataloaders(
        data_root="/neutrino/datasets/vww",
        dataset_name="vww",
        batch_size=128,
    )["test"]
    eval_fn = get_eval_function("mobilenetv3_large", "vww")
    ACC = eval_fn(model, test_loader)
    assert abs(ACC["acc"] - 0.891) < 0.001


@pytest.mark.slow
def test_mb3_small_vww():
    model = get_model(
        model_name="mobilenetv3_small",
        dataset_name="vww",
        pretrained=True,
    )
    test_loader = get_dataloaders(
        data_root="/neutrino/datasets/vww",
        dataset_name="vww",
        batch_size=128,
    )["test"]
    eval_fn = get_eval_function("mobilenetv3_small", "vww")
    ACC = eval_fn(model, test_loader)
    assert abs(ACC["acc"] - 0.892) < 0.001


@pytest.mark.slow
def test_yolov3_voc():
    model = get_model(
        model_name="yolo3",
        dataset_name="voc",
        pretrained=True,
    )
    eval_fn = get_eval_function("yolo3", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov3"
    )

    assert abs(APs["mAP"] - 0.829) < 0.001


@pytest.mark.slow
def test_yolov4s_voc():
    model = get_model(
        model_name="yolo4s",
        dataset_name="voc",
        pretrained=True,
    )
    eval_fn = get_eval_function("yolo4s", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4s"
    )

    assert abs(APs["mAP"] - 0.857) < 0.001

@pytest.mark.slow
def test_yolov4m_voc():
    model = get_model(
        model_name="yolo4m",
        dataset_name="voc",
        pretrained=True,
    )
    eval_fn = get_eval_function("yolo4m", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4m"
    )

    assert abs(APs["mAP"] - 0.882) < 0.001


@pytest.mark.slow
def test_yolov4l_voc():
    model = get_model(
        model_name="yolo4l",
        dataset_name="voc",
        pretrained=True,
    )
    eval_fn = get_eval_function("yolo4l", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolo4l"
    )

    assert abs(APs["mAP"] - 0.882) < 0.001


@pytest.mark.slow
def test_yolov4l_leaky_voc():
    model = get_model(
        model_name="yolo4l_leaky",
        dataset_name="voc",
        pretrained=True,
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
    model = get_model(
        model_name="yolo4x",
        dataset_name="voc",
        pretrained=True,
    )
    eval_fn = get_eval_function("yolo4x", "voc")
    APs = eval_fn(
        model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc", net="yolov4x"
    )

    assert abs(APs["mAP"] - 0.893) < 0.001


@pytest.mark.slow
def test_yolov5s_voc():
    model = get_model(
        model_name="yolo5_6s",
        dataset_name="voc",
        pretrained=True,
    )
    eval_fn = get_eval_function("yolo5_6s", "voc")
    APs = eval_fn(model, "/neutrino/datasets/VOCdevkit/VOC2007/", _set="voc")

    assert abs(APs["mAP"] - 0.821) < 0.001
