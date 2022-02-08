import unittest
import pytest
from pathlib import Path

from deeplite_torch_zoo import get_data_splits_by_name

DATASETS_ROOT = Path('/neutrino/datasets')


class TestDatasets(unittest.TestCase):
    def test_cifar100_dataset(self):
        BATCH_SIZE = 128
        datasplit = get_data_splits_by_name(
            dataset_name="cifar100", batch_size=BATCH_SIZE
        )
        train_len = len(datasplit["train"])
        test_len = len(datasplit["test"])
        self.assertEqual(train_len, 391)
        self.assertEqual(test_len, 79)

    def test_mnist_dataset(self):
        BATCH_SIZE = 128
        datasplit = get_data_splits_by_name(dataset_name="mnist", batch_size=BATCH_SIZE)
        train_len = len(datasplit["train"])
        test_len = len(datasplit["test"])
        self.assertEqual(train_len, 469)
        self.assertEqual(test_len, 79)

    @pytest.mark.skip(reason="The folder doesn't exist")
    def test_vww_dataset(self):
        BATCH_SIZE = 128
        datasplit = get_data_splits_by_name(
            dataset_name="vww",
            data_root=str(DATASETS_ROOT / "vww"),
            batch_size=BATCH_SIZE,
        )
        train_len = len(datasplit["train"])
        test_len = len(datasplit["test"])
        self.assertEqual(train_len, 901)
        self.assertEqual(test_len, 63)

    def test_imagenet16_dataset(self):
        BATCH_SIZE = 128
        datasplit = get_data_splits_by_name(
            dataset_name="imagenet",
            data_root=str(DATASETS_ROOT / "imagenet16"),
            batch_size=BATCH_SIZE,
        )
        train_len = len(datasplit["train"])
        test_len = len(datasplit["test"])
        self.assertEqual(train_len, 1408)
        self.assertEqual(test_len, 332)

    def test_imagenet10_dataset(self):
        BATCH_SIZE = 128
        datasplit = get_data_splits_by_name(
            data_root=str(DATASETS_ROOT / "imagenet10"),
            dataset_name="imagenet",
            batch_size=BATCH_SIZE,
        )
        train_len = len(datasplit["train"])
        test_len = len(datasplit["test"])
        self.assertEqual(train_len, 3010)
        self.assertEqual(test_len, 118)

    @pytest.mark.skip(reason="The folder doesn't exist")
    def test_imagenet1000_dataset(self):
        BATCH_SIZE = 128
        datasplit = get_data_splits_by_name(
            data_root=str(DATASETS_ROOT / "imagenet"),
            dataset_name="imagenet",
            batch_size=BATCH_SIZE,
        )
        train_len = len(datasplit["train"])
        test_len = len(datasplit["test"])
        self.assertEqual(train_len, 10010)
        self.assertEqual(test_len, 391)

    def test_voc0712_dataset(self):
        BATCH_SIZE = 128
        datasplit = get_data_splits_by_name(
            data_root=str(DATASETS_ROOT / "VOCdevkit"),
            dataset_name="voc",
            model_name="vgg16_ssd",
            batch_size=BATCH_SIZE,
        )
        train_len = len(datasplit["train"])
        test_len = len(datasplit["test"])
        self.assertEqual(train_len, 130)
        self.assertEqual(test_len, 39)

    def test_voc_yolo_dataset(self):
        BATCH_SIZE = 128
        datasplit = get_data_splits_by_name(
            data_root=str(DATASETS_ROOT / "VOCdevkit"),
            dataset_name="voc",
            model_name="yolo",
            batch_size=BATCH_SIZE,
        )
        train_len = len(datasplit["train"])
        test_len = len(datasplit["test"])
        self.assertEqual(train_len, 130)
        self.assertEqual(test_len, 39)

    def test_coco_yolo_dataset(self):
        BATCH_SIZE = 10
        datasplit = get_data_splits_by_name(
            data_root=str(DATASETS_ROOT / "coco"),
            dataset_name="coco",
            model_name="yolo",
            batch_size=BATCH_SIZE,
        )
        train_len = len(datasplit["train"])
        test_len = len(datasplit["test"])
        self.assertEqual(train_len, 11829)
        self.assertEqual(test_len, 500)


if __name__ == "__main__":
    unittest.main()
