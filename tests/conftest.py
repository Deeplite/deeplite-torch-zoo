import shutil
import contextlib
from abc import ABC, abstractmethod
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from deeplite_torch_zoo.api.datasets.utils import get_dataloader
from deeplite_torch_zoo import get_dataloaders


class FakeDataset(ABC, Dataset):
    def __init__(self, num_samples=100, num_classes=None, img_size=416, device="cuda"):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = device

    @abstractmethod
    def __getitem__(self):
        """Returns a fake sample that matches the original dataset samples, check the original dataset"""

    @abstractmethod
    def __len__(self):
        """Returns number of samples, by default 100"""


class VocYoloFake(FakeDataset):
    def __init__(self, num_samples=2, num_classes=None, img_size=224, device="cuda"):
        super(VocYoloFake, self).__init__(
            num_samples=num_samples,
            num_classes=num_classes,
            img_size=img_size,
            device=device,
        )

    def __getitem__(self, idx):
        img = torch.rand(3, self.img_size, self.img_size, dtype=torch.float32).to(
            self.device
        )
        bboxes = torch.zeros(2, 6)
        num_bboxes = bboxes.shape[0]
        img_id = 0
        return img, bboxes, num_bboxes, img_id

    def __len__(self):
        return self.num_samples

class SegmentationFake(FakeDataset):
    def __init__(self, num_samples=2, num_classes=None, img_size=224, device="cpu"):
        super(SegmentationFake, self).__init__(
            num_samples=num_samples,
            num_classes=num_classes,
            img_size=img_size,
            device=device,
        )
        self.class_names = ['mock'] * num_classes

    def __getitem__(self, idx):
        img = torch.rand(3, self.img_size, self.img_size, dtype=torch.float32).to(
            self.device
        )
        msk = torch.rand(self.img_size, self.img_size, dtype=torch.float32).to(
            self.device
        )
        return img, msk

    def __len__(self):
        return self.num_samples

    def untransform(self, img, mask):
        return img, mask


@pytest.fixture()
def mock_segmentation_dataloader():
    dataset = SegmentationFake(num_classes=21)
    dataloader = get_dataloader(dataset=dataset)
    yield dataloader


@pytest.fixture
def set_torch_seed_value():
    @contextlib.contextmanager
    def set_torch_seed(seed: int = 42):
        saved_seed = torch.seed()
        torch.manual_seed(seed)
        yield
        torch.manual_seed(saved_seed)

    yield set_torch_seed


@pytest.fixture
def imagewoof160_dataloaders(data_root='./', batch_size=32):
    p = Path(data_root)
    dataloaders = get_dataloaders(
        dataset_name='imagewoof_160',
        model_name='resnet18',
        data_root=data_root,
        batch_size=batch_size,
        map_to_imagenet_labels=True,
    )
    yield dataloaders
    (p / 'imagewoof160.zip').unlink()
    shutil.rmtree(p / 'imagewoof160')


@pytest.fixture
def cifar100_dataloaders(data_root='./', batch_size=32):
    p = Path(data_root)
    dataloaders = get_dataloaders(
        dataset_name='cifar100',
        model_name='resnet18',
        data_root=data_root,
        batch_size=batch_size,
    )
    yield dataloaders
    (p / 'cifar-100-python.tar.gz').unlink()
    shutil.rmtree(p / 'cifar-100-python')
