import torch
from mmcv.parallel import collate

from deeplite_torch_zoo.src.poseestimation.datasets import build_dataloader, build_dataset
from deeplite_torch_zoo.wrappers.eval.rcnn import get_cfg

__all__ = ["get_data_loaders"]


def _get_train_loader(cfg):
    train_dataset = build_dataset(cfg.data.train)
    dataloader_setting = dict(
        samples_per_gpu=cfg.data.get("samples_per_gpu", {}),
        workers_per_gpu=cfg.data.get("workers_per_gpu", {}),
        # cfg.gpus will be ignored if distributed
        num_gpus=1,  # len(cfg.gpu_ids),
        dist=False,
        seed=123,
    )
    dataloader_setting = dict(
        dataloader_setting, **cfg.data.get("train_dataloader", {})
    )

    data_loader = build_dataloader(train_dataset, **dataloader_setting)
    return data_loader


def _get_test_loader(cfg):
    test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    return data_loader


def get_data_loaders(dataset_type):
    cfg = get_cfg(_set=dataset_type)
    train_loader = _get_train_loader(cfg)
    test_loader = _get_test_loader(cfg)
    data_splits = {"train": train_loader, "test": test_loader}
    return data_splits
