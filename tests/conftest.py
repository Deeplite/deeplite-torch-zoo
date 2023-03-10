import contextlib
from abc import ABC, abstractmethod

import pytest
import torch
from torch.utils.data import Dataset

from deeplite_torch_zoo.utils import switch_train_mode

IMAGENETTE_IMAGENET_CLS_LABEL_MAP = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)
IMAGEWOOF_IMAGENET_CLS_LABEL_MAP =  (155, 159, 162, 167, 182, 193, 207, 229, 258, 273)


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


def imagenet_eval_fast(model, dataloader, device='cuda',
    top_k=5, label_map=IMAGEWOOF_IMAGENET_CLS_LABEL_MAP, max_iters=1):
    if not torch.cuda.is_available():
        device = 'cpu'

    model.to(device)
    pred = []
    targets = []
    with switch_train_mode(model, is_training=False):
        with torch.no_grad():
            for iter_no, (inputs, target) in enumerate(dataloader):
                if iter_no > max_iters:
                    break
                inputs = inputs.to(device)
                target = torch.tensor([label_map[label] for label in target])
                target = target.to(device)
                y = model(inputs)
                pred.append(y.argsort(1, descending=True)[:, :top_k])
                targets.append(target)

    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()

    return {'acc': top1, 'acc_top5': top5}


@pytest.fixture
def imagenet_eval_fast_fn():
    yield imagenet_eval_fast


@pytest.fixture
def set_torch_seed_value():
    @contextlib.contextmanager
    def set_torch_seed(seed: int = 42):
        saved_seed = torch.seed()
        torch.manual_seed(seed)
        yield
        torch.manual_seed(saved_seed)

    yield set_torch_seed