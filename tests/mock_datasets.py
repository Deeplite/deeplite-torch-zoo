from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["VocYoloFake", "SegmentationFake"]


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
    def __init__(self, num_samples=100, num_classes=None, img_size=416, device="cuda"):
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
    def __init__(self, num_samples=100, num_classes=None, img_size=416, device="cuda"):
        super(SegmentationFake, self).__init__(
            num_samples=num_samples,
            num_classes=num_classes,
            img_size=img_size,
            device=device,
        )

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
