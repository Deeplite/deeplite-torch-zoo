from pathlib import Path

import pytest

from deeplite_torch_zoo import get_data_splits_by_name

DATASETS_ROOT = Path('/neutrino/datasets')


def test_cifar100_dataset():
    BATCH_SIZE = 128
    datasplit = get_data_splits_by_name(
        data_root='./',
        model_name='resnet50',
        dataset_name="cifar100",
        batch_size=BATCH_SIZE
    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len == 391
    assert test_len ==  79


def test_cifar10_dataset():
    BATCH_SIZE = 128
    datasplit = get_data_splits_by_name(
        data_root='./',
        model_name='resnet50',
        dataset_name="cifar10",
        batch_size=BATCH_SIZE
    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len ==  391
    assert test_len ==  79


def test_mnist_dataset():
    BATCH_SIZE = 128
    datasplit = get_data_splits_by_name(
        data_root="./",
        dataset_name="mnist",
        model_name="lenet5_mnist",
        batch_size=BATCH_SIZE,

    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len ==  469
    assert test_len ==  79


@pytest.mark.local
def test_vww_dataset():
    BATCH_SIZE = 128
    datasplit = get_data_splits_by_name(
        dataset_name="vww",
        data_root=str(DATASETS_ROOT / "vww"),
        batch_size=BATCH_SIZE,
    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len ==  901
    assert test_len ==  63


@pytest.mark.local
def test_imagenet16_dataset():
    BATCH_SIZE = 128
    datasplit = get_data_splits_by_name(
        dataset_name="imagenet",
        data_root=str(DATASETS_ROOT / "imagenet16"),
        batch_size=BATCH_SIZE,
    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len ==  1408
    assert test_len ==  332


@pytest.mark.local
def test_imagenet10_dataset():
    BATCH_SIZE = 128
    datasplit = get_data_splits_by_name(
        data_root=str(DATASETS_ROOT / "imagenet10"),
        dataset_name="imagenet",
        batch_size=BATCH_SIZE,
    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len ==  3010
    assert test_len ==  118


@pytest.mark.local
def test_imagenet1000_dataset():
    BATCH_SIZE = 128
    datasplit = get_data_splits_by_name(
        data_root=str(DATASETS_ROOT / "imagenet"),
        dataset_name="imagenet",
        batch_size=BATCH_SIZE,
    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len ==  10010
    assert test_len ==  391


@pytest.mark.local
def test_voc0712_dataset():
    BATCH_SIZE = 128
    datasplit = get_data_splits_by_name(
        data_root=str(DATASETS_ROOT / "VOCdevkit"),
        dataset_name="voc",
        model_name="vgg16_ssd",
        batch_size=BATCH_SIZE,
    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len ==  130
    assert test_len ==  39


@pytest.mark.local
def test_voc_yolo_dataset():
    BATCH_SIZE = 128
    datasplit = get_data_splits_by_name(
        data_root=str(DATASETS_ROOT / "VOCdevkit"),
        dataset_name="voc",
        model_name="yolo",
        batch_size=BATCH_SIZE,
    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len ==  130
    assert test_len ==  39


@pytest.mark.local
def test_coco_yolo_dataset():
    BATCH_SIZE = 10
    datasplit = get_data_splits_by_name(
        data_root=str(DATASETS_ROOT / "coco"),
        dataset_name="coco",
        model_name="yolo",
        batch_size=BATCH_SIZE,
    )
    train_len = len(datasplit["train"])
    test_len = len(datasplit["test"])
    assert train_len ==  11829
    assert test_len ==  500
