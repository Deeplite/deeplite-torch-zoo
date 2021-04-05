import numpy as np
import torch
from torch.utils import data
from torchvision.datasets import VOCSegmentation


class Pascal_Data(VOCSegmentation):
    class_names = np.array(
        [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv/monitor",
        ]
    )

    def __init__(self, root, image_set="train", backbone="vgg", download=False):
        transform = self.transform
        self.backbone = backbone
        if backbone == "vgg":
            self.mean = np.array([104.00698793, 116.66876762, 122.67891434])  # BGR
            self.std = np.array([1.0, 1.0, 1.0])
        else:
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        # If not automatically download, please use the mirror link:
        # http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
        super(Pascal_Data, self).__init__(
            root, image_set=image_set, transforms=transform, download=download
        )

    def transform(self, img, lbl):
        if self.backbone == "vgg":
            img = np.array(img, dtype=np.uint8)
            lbl = np.array(lbl, dtype=np.uint8)
            img = img[:, :, ::-1]  # RGB -> BGR
            img = np.array(img, dtype=np.float64)
        else:
            img = np.array(img, dtype=np.float64)
            img /= 255.0
        lbl = np.array(lbl, dtype=np.float64)
        lbl[lbl == 255] = -1  # Ignore contour
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= self.std
        img += self.mean
        if self.backbone == "resnet":
            img *= 255
        img = img.astype(np.uint8)
        if self.backbone == "vgg":
            img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


def get_loader(opts):
    import os

    from data_loader import Pascal_Data

    kwargs = {"num_workers": 1} if "cuda" in str(opts.cuda) else {}
    if opts.mode in ["train", "demo"]:
        modes = ["train", "val"]
    else:
        modes = [opts.mode, opts.mode]
    download = not os.path.isdir(opts.root_dataset)
    train_loader = data.DataLoader(
        Pascal_Data(
            opts.root_dataset,
            download=download,
            image_set=modes[0],
            backbone=opts.backbone,
        ),
        batch_size=1,
        shuffle=True,
        **kwargs
    )
    val_loader = data.DataLoader(
        Pascal_Data(
            opts.root_dataset,
            download=download,
            image_set=modes[1],
            backbone=opts.backbone,
        ),
        batch_size=1,
        shuffle=False,
        **kwargs
    )
    return train_loader, val_loader
