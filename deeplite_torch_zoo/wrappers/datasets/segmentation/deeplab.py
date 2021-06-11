import sys

from ..utils import get_dataloader
from deeplite_torch_zoo.src.segmentation.deeplab.dataloaders.datasets import combine_dbs, pascal, sbd


__all__ = ["get_voc_for_deeplab_mobilenet"]


def get_voc_for_deeplab_mobilenet(
    data_root, sbd_root=None, batch_size=4, num_workers=4, img_size=512, fp16=False, distributed=False, device="cuda", **kwargs
):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_dataset = pascal.VOCSegmentation(
        data_root, base_size=img_size, crop_size=img_size, split="train"
    )
    test_dataset = pascal.VOCSegmentation(data_root, base_size=img_size, crop_size=img_size, split="val")
    if sbd_root:
        sbd_train = sbd.SBDSegmentation(
            sbd_root, base_size=img_size, crop_size=img_size, split=["train", "val"]
        )
        train_dataset = combine_dbs.CombineDBs([train_dataset, sbd_train], excluded=[test_dataset])

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "test": test_loader}
