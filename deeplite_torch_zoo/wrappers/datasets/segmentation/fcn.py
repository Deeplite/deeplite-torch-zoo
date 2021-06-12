from torch.utils.data import DataLoader
from deeplite_torch_zoo.src.segmentation.fcn.data_loader import Pascal_Data as FCN_Voc


__all__ = ["get_voc_for_fcn32"]


def get_voc_for_fcn32(data_root, num_workers=4, backbone="vgg", **kwargs):
    train_dataset = FCN_Voc(root=data_root, image_set="train", backbone=backbone)
    valid_dataset = FCN_Voc(root=data_root, image_set="val", backbone=backbone)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return {"train": train_loader, "test": val_loader}
