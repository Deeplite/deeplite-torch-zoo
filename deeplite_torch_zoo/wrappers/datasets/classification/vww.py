import os

import pyvww
from deeplite_torch_zoo.wrappers.datasets.classification.augs import \
    get_vanilla_transforms
from deeplite_torch_zoo.wrappers.datasets.utils import get_dataloader
from deeplite_torch_zoo.wrappers.registries import DATA_WRAPPER_REGISTRY

__all__ = ["get_vww"]


@DATA_WRAPPER_REGISTRY.register(dataset_name='vww')
def get_vww(data_root, batch_size=128, img_size=224, num_workers=4, fp16=False,
    distributed=False, device="cuda", **kwargs):

    if len(kwargs):
        import sys
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_transforms, val_transforms = get_vanilla_transforms(
        img_size,
    )

    train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=os.path.join(data_root, "all"),
        annFile=os.path.join(data_root, "annotations/instances_train.json"),
        transform=train_transforms,
    )

    test_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=os.path.join(data_root, "all"),
        annFile=os.path.join(data_root, "annotations/instances_val.json"),
        transform=val_transforms,
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "test": test_loader}
