from torch.utils.data import DataLoader

from deeplite_torch_zoo.src.segmentation.deeplab.dataloaders.datasets import (
    combine_dbs,
    pascal,
    sbd,
)


__all__ = ["get_voc_for_deeplab_mobilenet"]


def get_voc_for_deeplab_mobilenet(
    data_root, sbd_root=None, batch_size=4, num_workers=4, img_size=512, **kwargs
):
    train_set = pascal.VOCSegmentation(
        data_root, base_size=img_size, crop_size=img_size, split="train"
    )
    val_set = pascal.VOCSegmentation(data_root, base_size=img_size, crop_size=img_size, split="val")
    if sbd_root:
        sbd_train = sbd.SBDSegmentation(
            sbd_root, base_size=img_size, crop_size=img_size, split=["train", "val"]
        )
        train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "test": val_loader}
