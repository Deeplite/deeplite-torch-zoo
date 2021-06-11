import sys

from ..utils import get_dataloader
from deeplite_torch_zoo.src.segmentation.fcn.data_loader import Pascal_Data as FCN_Voc


__all__ = ["get_voc_for_fcn32"]


def get_voc_for_fcn32(data_root, batch_size=4, num_workers=4, backbone="vgg", fp16=False, distributed=False, device="cuda", **kwargs):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_dataset = FCN_Voc(root=data_root, image_set="train", backbone=backbone)
    test_dataset = FCN_Voc(root=data_root, image_set="val", backbone=backbone)

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "test": test_loader}
