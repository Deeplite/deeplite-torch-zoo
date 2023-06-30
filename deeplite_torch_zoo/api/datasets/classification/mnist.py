import os
from os.path import expanduser

import torch
import torchvision
from torchvision import transforms

from deeplite_torch_zoo.api.datasets.utils import create_loader
from deeplite_torch_zoo.api.registries import DATASET_WRAPPER_REGISTRY
from deeplite_torch_zoo.utils import LOGGER


__all__ = ["get_mnist"]

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


@DATASET_WRAPPER_REGISTRY.register(dataset_name='mnist')
def get_mnist(
    data_root=None,
    img_size=224,
    batch_size=64,
    test_batch_size=None,
    download=True,
    use_prefetcher=False,
    num_workers=1,
    distributed=False,
    pin_memory=False,
    device=torch.device('cuda'),
    mean=MNIST_MEAN,
    std=MNIST_STD,
    re_prob=0.0,
    re_mode='pixel',
    re_count=1,
    num_aug_repeats=0,
    no_aug=False,
    collate_fn=None,
    use_multi_epochs_loader=False,
    worker_seeding='all',
):
    if isinstance(device, str):
        device = torch.device(device)

    if data_root is None:
        data_root = os.path.join(expanduser('~'), '.deeplite-torch-zoo')

    dataset_train = torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        download=download,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
        ),
    )

    dataset_eval = torchvision.datasets.MNIST(
        root=data_root,
        train=False,
        download=download,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
        ),
    )

    train_loader = create_loader(
        dataset_train,
        input_size=img_size,
        batch_size=batch_size,
        is_training=True,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        num_aug_repeats=num_aug_repeats,
        mean=mean,
        std=std,
        num_workers=num_workers,
        distributed=distributed,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        device=device,
        use_multi_epochs_loader=use_multi_epochs_loader,
        worker_seeding=worker_seeding,
    )

    test_loader = create_loader(
        dataset_eval,
        input_size=img_size,
        batch_size=test_batch_size or batch_size,
        is_training=False,
        use_prefetcher=use_prefetcher,
        mean=mean,
        std=std,
        num_workers=num_workers,
        distributed=distributed,
        pin_memory=pin_memory,
        device=device,
    )

    return {'train': train_loader, 'test': test_loader}
