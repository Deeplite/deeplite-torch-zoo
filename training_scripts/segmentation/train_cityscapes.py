import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import torch
import albumentations as albu
from torch.utils.data import Dataset
from torch import Tensor
from pathlib import Path
from torchvision import io
import random
import torchvision.transforms.functional as TF 
from torch import nn
from typing import List, Union, Tuple
import math

class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask

class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h+1)
            x1 = random.randint(0, margin_w+1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]
        return img, mask
    
class Normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        img = img.float()
        img /= 255
        img = TF.normalize(img, self.mean, self.std)
        return img, mask
    
class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]

        # scale the image 
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        img = TF.resize(img, (alignH, alignW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (alignH, alignW), TF.InterpolationMode.NEAREST)
        return img, mask 
    
class Cityscapes(Dataset):
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 
                'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    ID2TRAINID = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255,
                  17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: -1}
    
    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.label_map = np.arange(256)
        for id, trainid in self.ID2TRAINID.items():
            self.label_map[id] = trainid

        img_path = Path(root) / 'leftImg8bit' / split
        self.files = list(img_path.rglob('*.png'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('leftImg8bit', 'gtFine').replace('.png', '_labelIds.png')

        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        
        if self.transform:
            image, label = self.transform(image, label)

        return image, self.encode(label.squeeze().numpy()).long()

    def encode(self, label: Tensor) -> Tensor:
        label = self.label_map[label]
        return torch.from_numpy(label)


ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda:1'

class CrossEntropy(nn.Module):
    __name__ = "CE"
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)
            
class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
            
transform_crop = Compose([
    Normalize(),
])
transform_full = Compose([
    Normalize(),
])
            
    

def compute_mIOU(outputs, targets, num_classes):
    """
    Compute mean Intersection over Union (mIOU) for Cityscapes dataset
    """
    with torch.no_grad():
        confusion_matrix = torch.zeros(num_classes, num_classes)
        for output, target in zip(outputs, targets):
            mask = target != 255  # ignore void class
            output = torch.argmax(output, dim=1)[mask].squeeze()
            target = target[mask].squeeze()
            confusion_matrix += torch.bincount(num_classes * target.long() + output.long(), minlength=num_classes**2).reshape(num_classes, num_classes)
        
        intersection = torch.diag(confusion_matrix)
        union = torch.sum(confusion_matrix, dim=0) + torch.sum(confusion_matrix, dim=1) - intersection
        
        iou = intersection / union
        miou = torch.mean(iou)
    
    return miou.item()

# create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=20, 
    in_channels = 3,
)
    
train_data = Cityscapes('/neutrino/datasets/cityscapes/', split='train', transform=transform_crop)
valid_data = Cityscapes('/neutrino/datasets/cityscapes/', split='val', transform=transform_full)
        
train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=16, drop_last=True)                
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=2, shuffle=False, num_workers=16, drop_last=True)                

loss = CrossEntropy()
metrics = [compute_mIOU]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

max_score = 0

for i in range(0, 60):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    torch.save(model, './best_model_cit.pth')

    if i == 35:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
