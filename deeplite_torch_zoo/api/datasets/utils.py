from functools import partial

import torch

from timm.data.dataset import IterableImageDataset
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from timm.data.loader import fast_collate, PrefetchLoader, MultiEpochsDataLoader, _worker_init

from deeplite_torch_zoo.src.classification.augmentations.augs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def create_loader(
    dataset,
    input_size,
    batch_size,
    is_training=False,
    use_prefetcher=True,
    no_aug=False,
    re_prob=0.,
    re_mode='const',
    re_count=1,
    re_num_splits=0,
    num_aug_repeats=0,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    num_workers=1,
    distributed=False,
    collate_fn=None,
    pin_memory=False,
    fp16=False,  # deprecated, use img_dtype
    img_dtype=torch.float32,
    device=torch.device('cuda'),
    use_multi_epochs_loader=False,
    persistent_workers=True,
    worker_seeding='all',
):
    if isinstance(input_size, int):
        input_size = (3, input_size, input_size)

    if isinstance(dataset, IterableImageDataset):
        # give Iterable datasets early knowledge of num_workers so that sample estimates
        # are correct before worker processes are launched
        dataset.set_loader_cfg(num_workers=num_workers)

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_size[0],
            device=device,
            fp16=fp16,  # deprecated, use img_dtype
            img_dtype=img_dtype,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader
