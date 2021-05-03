import unittest

import pytest
import torch

from deeplite_torch_zoo.wrappers.wrapper import get_model_by_name, get_data_splits_by_name, get_models_names_for
from tests.fake_tests.Datasets.FakeDataset import *


class TestModelsFake(unittest.TestCase):
    _random_mnist_input_tensor = torch.randn(1, 1, 28, 28)
    _random_cifar100_input_tensor = torch.randn(1, 3, 32, 32)
    _random_imagenet_input_tensor = torch.randn(3, 3, 224, 224)

    @pytest.mark.test_torchvision_models_imagenet
    def test_torchvision_models_imagenet(self):
        model_names = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'alexnet', 'vgg16', 'squeezenet1_0',
            'densenet161', 'densenet201', 'googlenet', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'mobilenet_v2',
            'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'mnasnet1_0', 'mnasnet0_5',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'q_resnet18', 'q_resnet50',
            'q_googlenet', 'q_shufflenet_v2_x0_5', 'q_shufflenet_v2_x1_0', 'q_mobilenet_v2', 'q_resnext101_32x8d',
            #'q_inception_v3' # Commented for faster testing
        ]
        for model_name in model_names:
            model = get_model_by_name(
                model_name=model_name,
                dataset_name="imagenet",
                pretrained=True,
                progress=False,
                device="cpu",
            )
            y = model(self._random_imagenet_input_tensor)
            self.assertEqual(y.shape, (3, 1000))

    def test_resnet18_imagenet16(self):
        model = get_model_by_name(
            model_name="resnet18",
            dataset_name="imagenet16",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_imagenet_input_tensor)
        self.assertEqual(y.shape, (3, 16))

    def test_resnet50_imagenet16(self):
        model = get_model_by_name(
            model_name="resnet50",
            dataset_name="imagenet16",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_imagenet_input_tensor)
        self.assertEqual(y.shape, (3, 16))

    def test_resnet18_imagenet10(self):
        model = get_model_by_name(
            model_name="resnet18",
            dataset_name="imagenet10",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_imagenet_input_tensor)
        self.assertEqual(y.shape, (3, 10))

    # def test_mobilenet_v2_1_0_imagenet10(self):
    #     model = imagenet10.mobilenet_v2_1_0(pretrained=True, progress=False)
    #     y = model(self._random_imagenet_input_tensor)
    #     self.assertEqual(y.shape, (3, 10))

    def test_mobilenet_v2_0_35_imagenet10(self):
        model = get_model_by_name(
            model_name="mobilenet_v2_0_35",
            dataset_name="imagenet10",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_imagenet_input_tensor)
        self.assertEqual(y.shape, (3, 10))

    def test_resnet18_vww(self):
        model = get_model_by_name(
            model_name="resnet18",
            dataset_name="vww",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_imagenet_input_tensor)
        self.assertEqual(y.shape, (3, 2))

    def test_resnet50_vww(self):
        model = get_model_by_name(
            model_name="resnet50",
            dataset_name="vww",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_imagenet_input_tensor)
        self.assertEqual(y.shape, (3, 2))


    def test_mobilenetv1_vww(self):
        model = get_model_by_name(
            model_name="mobilenet_v1",
            dataset_name="vww",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_imagenet_input_tensor)
        self.assertEqual(y.shape, (3, 2))

    def test_vgg19_cifar100(self):
        model = get_model_by_name(
            model_name="vgg19",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_resnet18_cifar100(self):
        model = get_model_by_name(
            model_name="resnet18",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_resnet50_cifar100(self):
        model = get_model_by_name(
            model_name="resnet50",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_densenet121_cifar100(self):
        model = get_model_by_name(
            model_name="densenet121",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_shufflenet_v2_larg_cifar100(self):
        model = get_model_by_name(
            model_name="shufflenet_v2_1_0",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_mobilenet_v1_cifar100(self):
        model = get_model_by_name(
            model_name="mobilenet_v1",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_mobilenet_v2_cifar100(self):
        model = get_model_by_name(
            model_name="mobilenet_v2",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_googlenet_cifar100(self):
        model = get_model_by_name(
            model_name="googlenet",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_pre_act_resnet18_cifar100(self):
        model = get_model_by_name(
            model_name="pre_act_resnet18",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_resnext29_2x64d_cifar100(self):
        model = get_model_by_name(
            model_name="resnext29_2x64d",
            dataset_name="cifar100",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_cifar100_input_tensor)
        self.assertEqual(y.shape, (1, 100))

    def test_lenet_mnist(self):
        model = get_model_by_name(
            model_name="lenet5",
            dataset_name="mnist",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_mnist_input_tensor)
        self.assertEqual(y.shape, (1, 10))

    def test_mlp2_mnist(self):
        model = get_model_by_name(
            model_name="mlp2",
            dataset_name="mnist",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_mnist_input_tensor)
        self.assertEqual(y.shape, (1, 10))

    def test_mlp4_mnist(self):
        model = get_model_by_name(
            model_name="mlp4",
            dataset_name="mnist",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_mnist_input_tensor)
        self.assertEqual(y.shape, (1, 10))

    @pytest.mark.test_mlp8_mnist
    def test_mlp8_mnist(self):
        model = get_model_by_name(
            model_name="mlp8",
            dataset_name="mnist",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        y = model(self._random_mnist_input_tensor)
        self.assertEqual(y.shape, (1, 10))

    @pytest.mark.test_mb1_ssd_voc_20_fake
    def test_mb1_ssd_voc_20_fake(self):
        model = get_model_by_name(
            model_name="mb1_ssd",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        train_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="mb1_ssd",
            batch_size=1,
            num_workers=0,
            device="cpu",
        )["train"]
        imgs = []
        for data in train_loader:
            imgs, _, _ = data
            break
        y1, y2 = model(imgs)
        self.assertEqual(list(y1.shape), [1, 3000, 21])
        self.assertEqual(list(y2.shape), [1, 3000, 4])

    @pytest.mark.test_mb2_ssd_lite_voc_20_fake
    def test_mb2_ssd_lite_voc_20_fake(self):
        model = get_model_by_name(
            model_name="mb2_ssd_lite",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        train_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="mb2_ssd_lite",
            batch_size=2,
            num_workers=0,
            device="cpu",
        )["train"]
        imgs = []
        for data in train_loader:
            imgs, _, _ = data
            break
        y1, y2 = model(imgs)
        self.assertEqual(list(y1.shape), [2, 3000, 21])
        self.assertEqual(list(y2.shape), [2, 3000, 4])

    @pytest.mark.test_mb2_ssd_lite_voc_1_fake
    def test_mb2_ssd_lite_voc_1_fake(self):
        model = get_model_by_name(
            model_name="mb2_ssd_lite",
            dataset_name="voc_1",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        train_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="mb2_ssd_lite",
            batch_size=2,
            num_workers=0,
            num_classes=1,
            device="cpu"
        )["train"]
        imgs = []
        for data in train_loader:
            imgs, _, _ = data
            break
        y1, y2 = model(imgs)
        self.assertEqual(list(y1.shape), [2, 3000, 2])
        self.assertEqual(list(y2.shape), [2, 3000, 4])

    @pytest.mark.test_mb2_ssd_lite_voc_2_fake
    def test_mb2_ssd_lite_voc_2_fake(self):
        model = get_model_by_name(
            model_name="mb2_ssd_lite",
            dataset_name="voc_2",
            pretrained=True,
            progress=False,
            device="cpu",
        )

        train_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="mb2_ssd_lite",
            batch_size=2,
            num_workers=0,
            num_classes=2,
            device="cpu",
        )["train"]
        imgs = []
        for data in train_loader:
            imgs, _, _ = data
            break
        y1, y2 = model(imgs)
        self.assertEqual(list(y1.shape), [2, 3000, 3])
        self.assertEqual(list(y2.shape), [2, 3000, 4])

    @pytest.mark.test_mb2_ssd_voc_20_fake
    def test_mb2_ssd_voc_20_fake(self):
        model = get_model_by_name(
            model_name="mb2_ssd",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        train_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="mb2_ssd_lite",
            batch_size=2,
            num_workers=0,
            num_classes=2,
            device="cpu",
        )["train"]
        imgs = []
        for data in train_loader:
            imgs, _, _ = data
            break
        y1, y2 = model(imgs)
        self.assertEqual(list(y1.shape), [2, 3000, 21])
        self.assertEqual(list(y2.shape), [2, 3000, 4])

    @pytest.mark.test_yolov3_voc_20_fake
    def test_yolov3_voc_20_fake(self):
        model = get_model_by_name(
            model_name="yolo3",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="yolo",
            batch_size=1,
            num_workers=0,
            num_classes=21,
            img_size=416,
            device="cpu",
        )["test"]
        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y[0][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[0][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[0][2].shape), [1, 13, 13, 3, 25])

        self.assertEqual(list(y[1][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[1][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[1][2].shape), [1, 13, 13, 3, 25])

    @pytest.mark.test_yolov3_voc_1_fake
    def test_yolov3_voc_1_fake(self):
        model = get_model_by_name(
            model_name="yolo3",
            dataset_name="voc_1",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="yolo",
            batch_size=1,
            num_workers=0,
            num_classes=1,
            img_size=416,
            device="cpu",
        )["test"]
        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y[0][0].shape), [1, 52, 52, 3, 6])
        self.assertEqual(list(y[0][1].shape), [1, 26, 26, 3, 6])
        self.assertEqual(list(y[0][2].shape), [1, 13, 13, 3, 6])

        self.assertEqual(list(y[1][0].shape), [1, 52, 52, 3, 6])
        self.assertEqual(list(y[1][1].shape), [1, 26, 26, 3, 6])
        self.assertEqual(list(y[1][2].shape), [1, 13, 13, 3, 6])

    @pytest.mark.test_yolov3_voc_2_fake
    def test_yolov3_voc_2_fake(self):
        model = get_model_by_name(
            model_name="yolo3",
            dataset_name="voc_2",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="yolo",
            batch_size=1,
            num_workers=0,
            num_classes=2,
            img_size=416,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y[0][0].shape), [1, 52, 52, 3, 7])
        self.assertEqual(list(y[0][1].shape), [1, 26, 26, 3, 7])
        self.assertEqual(list(y[0][2].shape), [1, 13, 13, 3, 7])

        self.assertEqual(list(y[1][0].shape), [1, 52, 52, 3, 7])
        self.assertEqual(list(y[1][1].shape), [1, 26, 26, 3, 7])
        self.assertEqual(list(y[1][2].shape), [1, 13, 13, 3, 7])

    @pytest.mark.test_yolo3_lisa_11_fake
    def test_yolo3_lisa_11_fake(self):
        model = get_model_by_name(
            model_name="yolo3",
            dataset_name="lisa_11",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        dataset = VocYoloFake(num_samples=1, num_classes=12, device="cpu")
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y[0][0].shape), [1, 52, 52, 3, 16])
        self.assertEqual(list(y[0][1].shape), [1, 26, 26, 3, 16])
        self.assertEqual(list(y[0][2].shape), [1, 13, 13, 3, 16])

        self.assertEqual(list(y[1][0].shape), [1, 52, 52, 3, 16])
        self.assertEqual(list(y[1][1].shape), [1, 26, 26, 3, 16])
        self.assertEqual(list(y[1][2].shape), [1, 13, 13, 3, 16])

    @pytest.mark.test_yolo4s_voc_20_fake
    def test_yolo4s_voc_20_fake(self):
        model = get_model_by_name(
            model_name="yolo4s",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="yolo",
            batch_size=1,
            num_workers=0,
            num_classes=21,
            img_size=416,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y[0][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[0][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[0][2].shape), [1, 13, 13, 3, 25])

        self.assertEqual(list(y[1][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[1][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[1][2].shape), [1, 13, 13, 3, 25])

    @pytest.mark.test_yolo4m_voc_20_fake
    def test_yolo4m_voc_20_fake(self):
        model = get_model_by_name(
            model_name="yolo4m",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="yolo",
            batch_size=1,
            num_workers=0,
            num_classes=21,
            img_size=416,
            device="cpu",
        )["test"]
        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y[0][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[0][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[0][2].shape), [1, 13, 13, 3, 25])

        self.assertEqual(list(y[1][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[1][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[1][2].shape), [1, 13, 13, 3, 25])

    @pytest.mark.test_yolo4l_voc_20_fake
    def test_yolo4l_voc_20_fake(self):
        model = get_model_by_name(
            model_name="yolo4l",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="yolo",
            batch_size=1,
            num_workers=0,
            num_classes=21,
            img_size=416,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y[0][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[0][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[0][2].shape), [1, 13, 13, 3, 25])

        self.assertEqual(list(y[1][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[1][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[1][2].shape), [1, 13, 13, 3, 25])

    @pytest.mark.test_yolo4x_voc_20_fake
    def test_yolo4x_voc_20_fake(self):
        model = get_model_by_name(
            model_name="yolo4x",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="yolo",
            batch_size=1,
            num_workers=0,
            num_classes=21,
            img_size=416,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y[0][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[0][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[0][2].shape), [1, 13, 13, 3, 25])

        self.assertEqual(list(y[1][0].shape), [1, 52, 52, 3, 25])
        self.assertEqual(list(y[1][1].shape), [1, 26, 26, 3, 25])
        self.assertEqual(list(y[1][2].shape), [1, 13, 13, 3, 25])

    @pytest.mark.test_ssd300_resnet18_voc_20_fake
    def test_ssd300_resnet18_voc_20_fake(self):
        model = get_model_by_name(
            model_name="ssd300_resnet18",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="ssd300",
            batch_size=1,
            num_workers=0,
            num_classes=21,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        model.eval()
        y1, y2 = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y1.shape), [1, 4, 8732])
        self.assertEqual(list(y2.shape), [1, 21, 8732])

    @pytest.mark.test_ssd300_resnet34_voc_20_fake
    def test_ssd300_resnet34_voc_20_fake(self):
        model = get_model_by_name(
            model_name="ssd300_resnet34",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="ssd300",
            batch_size=1,
            num_workers=0,
            num_classes=21,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        model.eval()
        y1, y2 = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y1.shape), [1, 4, 8732])
        self.assertEqual(list(y2.shape), [1, 21, 8732])

    @pytest.mark.test_ssd300_resnet50_voc_20_fake
    def test_ssd300_resnet50_voc_20_fake(self):
        model = get_model_by_name(
            model_name="ssd300_resnet50",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="ssd300",
            batch_size=1,
            num_workers=0,
            num_classes=21,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        model.eval()
        y1, y2 = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y1.shape), [1, 4, 8732])
        self.assertEqual(list(y2.shape), [1, 21, 8732])

    @pytest.mark.test_ssd300_vgg16_voc_20_fake
    def test_ssd300_vgg16_voc_20_fake(self):
        model = get_model_by_name(
            model_name="ssd300_vgg16",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/VOCdevkit",
            dataset_name="voc",
            model_name="ssd300",
            batch_size=1,
            num_workers=0,
            num_classes=21,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, _, _, _ = dataset[0]
        model.eval()
        y1, y2 = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y1.shape), [1, 4, 8732])
        self.assertEqual(list(y2.shape), [1, 21, 8732])

    @pytest.mark.test_unet_carvana_fake
    def test_unet_carvana_fake(self):
        model = get_model_by_name(
            model_name="unet",
            dataset_name="carvana",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/carvana",
            dataset_name="carvana",
            model_name="unet",
            num_workers=0,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, msk, _ = dataset[0]
        model.eval()
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y.shape), [1, *msk.shape])

    @pytest.mark.test_unet_scse_resnet18_voc_20_fake
    def test_unet_scse_resnet18_voc_20_fake(self):
        model = get_model_by_name(
            model_name="unet_scse_resnet18",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets",
            dataset_name="voc",
            model_name="unet",
            num_workers=0,
            device="cpu",
        )["test"]

        dataset = test_loader.dataset
        img, msk, _ = dataset[0]
        model.eval()
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y.shape), [1, 21, *msk.shape])

    @pytest.mark.test_unet_scse_resnet18_voc_1_fake
    def test_unet_scse_resnet18_voc_1_fake(self):
        model = get_model_by_name(
            model_name="unet_scse_resnet18",
            dataset_name="voc_1",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets",
            dataset_name="voc",
            model_name="unet",
            num_workers=0,
            num_classes=2,
            device="cpu",
        )["test"]
        dataset = test_loader.dataset
        img, msk, _ = dataset[0]
        model.eval()
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y.shape), [1, 1, *msk.shape])

    @pytest.mark.test_unet_scse_resnet18_voc_2_fake
    def test_unet_scse_resnet18_voc_2_fake(self):
        model = get_model_by_name(
            model_name="unet_scse_resnet18",
            dataset_name="voc_2",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets",
            dataset_name="voc",
            model_name="unet",
            num_workers=0,
            num_classes=3,
            device="cpu"
        )["test"]
        dataset = test_loader.dataset
        img, msk, _ = dataset[0]
        model.eval()
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y.shape), [1, 3, *msk.shape])

    @pytest.mark.test_unet_scse_resnet18_carvana_fake
    def test_unet_scse_resnet18_carvana_fake(self):
        model = get_model_by_name(
            model_name="unet_scse_resnet18",
            dataset_name="carvana",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets/carvana",
            dataset_name="carvana",
            model_name="unet",
            num_workers=0,
            device="cpu",
        )["test"]
        dataset = test_loader.dataset
        img, msk, _ = dataset[0]
        model.eval()
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y.shape), [1, *msk.shape])

    @pytest.mark.test_fcn32_voc_20_fake
    def test_fcn32_voc_20_fake(self):
        model = get_model_by_name(
            model_name="fcn32",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets",
            dataset_name="voc",
            model_name="fcn32",
            num_workers=0,
            backbone="vgg",
        )["test"]

        dataset = test_loader.dataset
        img, msk = dataset[0]
        model.eval()
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y.shape), [1, 21, *msk.shape])

    @pytest.mark.test_deeplab_mobilenet_voc_20_fake
    def test_deeplab_mobilenet_voc_20_fake(self):
        model = get_model_by_name(
            model_name="deeplab_mobilenet",
            dataset_name="voc_20",
            pretrained=True,
            progress=False,
            device="cpu",
        )
        test_loader = get_data_splits_by_name(
            data_root="fixture/datasets",
            dataset_name="voc",
            model_name="deeplab_mobilenet",
            num_workers=0,
            backbone="vgg",
            device="cpu",
        )["test"]
        dataset = test_loader.dataset
        img, msk = dataset[0]
        model.eval()
        y = model(torch.unsqueeze(img, dim=0))
        self.assertEqual(list(y.shape), [1, 21, *msk.shape])

