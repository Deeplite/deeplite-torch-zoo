import os
import sys
from os.path import expanduser

import torchvision
from torchvision import transforms

from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY
from deeplite_torch_zoo.utils import LOGGER

__all__ = ['get_cifar100', 'get_cifar10']


CIFAR_IMAGE_SIZE = 32


def _get_cifar(
    cifar_cls,
    data_root=None,
    batch_size=128,
    test_batch_size=None,
    num_workers=4,
    fp16=False,
    download=True,
    distributed=False,
    train_transforms=None,
    val_transforms=None,
):
    if data_root is None:
        data_root = os.path.join(expanduser('~'), '.deeplite-torch-zoo')

    default_train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(CIFAR_IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    default_val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_transforms = (
        train_transforms if train_transforms is not None else default_train_transforms
    )
    val_transforms = (
        val_transforms if val_transforms is not None else default_val_transforms
    )

    train_dataset = cifar_cls(
        root=data_root,
        train=True,
        download=download,
        transform=train_transforms,
    )

    test_dataset = cifar_cls(
        root=data_root,
        train=False,
        download=download,
        transform=val_transforms,
    )

    train_loader = create_loader(
        train_dataset,
        CIFAR_IMAGE_SIZE,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        is_training=True,
    )

    test_loader = create_loader(
        test_dataset,
        CIFAR_IMAGE_SIZE,
        batch_size=batch_size if test_batch_size is None else test_batch_size,
        num_workers=num_workers,
        distributed=distributed,
        is_training=True,
    )

    return {'train': train_loader, 'test': test_loader}


@DATASET_WRAPPER_REGISTRY.register(dataset_name='cifar100')
def get_cifar100(
    data_root=None,
    batch_size=128,
    test_batch_size=None,
    num_workers=4,
    fp16=False,
    download=True,
    distributed=False,
    train_transforms=None,
    val_transforms=None,
    **kwargs,
):
    if kwargs:
        LOGGER.warning(
            f'Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!'
        )

    cifar_cls = torchvision.datasets.CIFAR100
    return _get_cifar(
        cifar_cls,
        data_root=data_root,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        fp16=fp16,
        download=download,
        distributed=distributed,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )


@DATASET_WRAPPER_REGISTRY.register(dataset_name='cifar10')
def get_cifar10(
    data_root=None,
    batch_size=128,
    test_batch_size=None,
    num_workers=4,
    fp16=False,
    download=True,
    distributed=False,
    train_transforms=None,
    val_transforms=None,
    **kwargs,
):
    if kwargs:
        LOGGER.warning(
            f'Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!'
        )

    cifar_cls = torchvision.datasets.CIFAR10
    return _get_cifar(
        cifar_cls,
        data_root=data_root,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        fp16=fp16,
        download=download,
        distributed=distributed,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )
