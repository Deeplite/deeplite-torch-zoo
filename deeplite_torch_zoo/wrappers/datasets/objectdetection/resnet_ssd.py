import sys

from deeplite_torch_zoo.src.objectdetection.ssd.config.vgg_ssd_config import VGG_CONFIG as CONFIG
from deeplite_torch_zoo.wrappers.datasets.objectdetection.ssd import get_voc_for_ssd


__all__ = ["get_voc_for_resnet18_ssd", "get_voc_for_resnet34_ssd", "get_voc_for_resnet50_ssd"]


def get_voc_for_resnet18_ssd(data_root, batch_size=32, num_workers=4, num_classes=21, fp16=False,
    distributed=False, device="cuda", **kwargs):

    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    return get_voc_for_ssd(
        data_root=data_root, config=CONFIG(), num_classes=num_classes, batch_size=batch_size,
        num_workers=num_workers, fp16=fp16, distributed=distributed, device=device
    )


def get_voc_for_resnet34_ssd(data_root, batch_size=32, num_workers=4, num_classes=21, fp16=False,
    distributed=False, device="cuda", **kwargs):

    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    return get_voc_for_ssd(
        data_root=data_root, config=CONFIG(), num_classes=num_classes, batch_size=batch_size,
        num_workers=num_workers, fp16=fp16, distributed=distributed, device=device
    )


def get_voc_for_resnet50_ssd(data_root, batch_size=32, num_workers=4, num_classes=21, fp16=False,
    distributed=False, device="cuda", **kwargs):

    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    return get_voc_for_ssd(
        data_root=data_root, config=CONFIG(), num_classes=num_classes, batch_size=batch_size,
        num_workers=num_workers, fp16=fp16, distributed=distributed, device=device
    )
