import os

from ..utils import get_dataloader
from torchvision import datasets, transforms
from deeplite_torch_zoo.wrappers.registries import DATA_WRAPPER_REGISTRY


__all__ = ["get_imagenet"]

@DATA_WRAPPER_REGISTRY.register(dataset_name="imagenet16")
@DATA_WRAPPER_REGISTRY.register(dataset_name="imagenet10")
@DATA_WRAPPER_REGISTRY.register(dataset_name="imagenet")
def get_imagenet(data_root, batch_size=128, num_workers=4, fp16=False, distributed=False, device="cuda", **kwargs):

    if len(kwargs):
        import sys
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, "imagenet_training"),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(data_root, "imagenet_val"),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=not distributed, device=device)

    test_loader = get_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers,
        fp16=fp16, distributed=distributed, shuffle=False, device=device)

    return {"train": train_loader, "test": test_loader}
