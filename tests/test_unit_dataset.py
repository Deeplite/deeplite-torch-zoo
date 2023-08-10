from pathlib import Path, PurePosixPath
from unittest import mock
from unittest.mock import mock_open

import pytest

from deeplite_torch_zoo.api.datasets.classification.cifar import (
    get_cifar10, get_cifar100)
from deeplite_torch_zoo.api.datasets.classification.flowers102 import \
    get_flowers102
from deeplite_torch_zoo.api.datasets.classification.food101 import \
    get_food101
from deeplite_torch_zoo.api.datasets.classification.imagenet import \
    get_imagenet
from deeplite_torch_zoo.api.datasets.classification.imagenette import (
    get_imagenette_160, get_imagenette_320)
from deeplite_torch_zoo.api.datasets.classification.imagewoof import (
    get_imagewoof_160, get_imagewoof_320)
from deeplite_torch_zoo.api.datasets.classification.mnist import get_mnist
from deeplite_torch_zoo.api.datasets.classification.tiny_imagenet import \
    get_tinyimagenet
from deeplite_torch_zoo.api.datasets.classification.vww import get_vww


@mock.patch("deeplite_torch_zoo.api.datasets.classification.cifar._get_cifar")
def test_unit_cifar(*args):
    kwargs = dict(x=1, y=3)
    get_cifar100(**kwargs)
    get_cifar10(**kwargs)


class MockedPath(Path):
    _flavour = PurePosixPath._flavour
    rval = False
    def exists(self):
        return self.rval

    def is_dir(self):
        return self.rval


def get_dataloader(dataset, *args, **kwargs):
    return dataset


@mock.patch("deeplite_torch_zoo.api.datasets.classification.flowers102.Path", MockedPath)
@mock.patch("deeplite_torch_zoo.api.datasets.classification.flowers102.PIL.Image")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.flowers102.check_integrity", return_value=True)
@mock.patch("deeplite_torch_zoo.api.datasets.classification.flowers102.download_and_extract_archive")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.flowers102.download_url")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.flowers102.get_dataloader", get_dataloader)
@mock.patch("scipy.io.loadmat")
def test_unit_flower102(*args):
    kwargs = dict(x=1, y=3)

    MockedPath.rval = False
    with pytest.raises(RuntimeError):
        flower102 = get_flowers102(**kwargs)

    MockedPath.rval = True
    flower102 = get_flowers102(**kwargs)['train']
    flower102._image_files = list(range(10))
    flower102._labels = list(range(10))
    flower102.transform = mock.MagicMock(side_effect=lambda x: x)
    flower102.target_transform = mock.MagicMock(side_effect=lambda x: x)
    _, label = flower102[0]
    assert label == 0


@mock.patch("deeplite_torch_zoo.api.datasets.classification.food101.json")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.food101.Path", MockedPath)
@mock.patch("deeplite_torch_zoo.api.datasets.classification.food101.PIL.Image")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.food101.download_and_extract_archive")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.food101.get_dataloader", get_dataloader)
@mock.patch("builtins.open", new_callable=mock_open, read_data="data")
def test_unit_food101(*args):
    kwargs = dict(x=1, y=3)

    MockedPath.rval = False
    with pytest.raises(RuntimeError):
        food101 = get_food101(**kwargs)

    MockedPath.rval = True
    food101 = get_food101(**kwargs)['train']
    food101._image_files = list(range(10))
    food101._labels = list(range(10))
    food101.transform = mock.MagicMock(side_effect=lambda x: x)
    food101.target_transform = mock.MagicMock(side_effect=lambda x: x)
    _, label = food101[0]
    assert label == 0


@mock.patch("deeplite_torch_zoo.api.datasets.classification.imagenet.datasets")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.imagenet.get_dataloader", get_dataloader)
def test_unit_imagenet(*args):
    get_imagenet('', x=3)


@mock.patch("deeplite_torch_zoo.api.datasets.classification.food101.PIL.Image")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.imagenette.Path", MockedPath)
@mock.patch("deeplite_torch_zoo.api.datasets.classification.imagenette.download_and_extract_archive")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.imagenette.verify_str_arg")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.imagenette.get_dataloader", get_dataloader)
@mock.patch("deeplite_torch_zoo.api.datasets.classification.imagenette.os.scandir")
def test_unit_imagenette(*args):
    MockedPath.rval = True
    get_imagenette_160('', x=3)
    imagenette = get_imagenette_320('', x=3)['train']
    imagenette._image_files = list(range(10))
    imagenette._labels = list(range(10))
    imagenette.transform = mock.MagicMock(side_effect=lambda x: x)
    imagenette.target_transform = mock.MagicMock(side_effect=lambda x: x)
    _, label = imagenette[0]
    assert label == 0


@mock.patch("deeplite_torch_zoo.api.datasets.classification.imagewoof.Imagewoof")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.imagewoof.get_dataloader", get_dataloader)
def test_unit_imagewoof(*args):
    get_imagewoof_160('', x=3)
    get_imagewoof_320('', x=3)


@mock.patch("deeplite_torch_zoo.api.datasets.classification.vww.VisualWakeWordsClassification")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.vww.get_dataloader", get_dataloader)
def test_unit_vww(*args):
    get_vww('', x=3)


@mock.patch("deeplite_torch_zoo.api.datasets.classification.mnist.torchvision")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.mnist.get_dataloader", get_dataloader)
def test_unit_mnist(*args):
    get_mnist('', x=1, y=2)


@mock.patch("deeplite_torch_zoo.api.datasets.classification.tiny_imagenet.datasets")
@mock.patch("deeplite_torch_zoo.api.datasets.classification.tiny_imagenet.get_dataloader", get_dataloader)
def test_unit_tinyimagenet(*args):
    get_tinyimagenet('', x=1, y=2)
