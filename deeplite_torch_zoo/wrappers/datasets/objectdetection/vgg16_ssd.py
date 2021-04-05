from deeplite_torch_zoo.src.objectdetection.mb_ssd.config.vgg_ssd_config import VGG_CONFIG
from deeplite_torch_zoo.wrappers.datasets.objectdetection.ssd import get_voc_for_ssd


__all__ = ["get_voc_for_vgg16_ssd"]


def get_voc_for_vgg16_ssd(data_root, batch_size=32, num_workers=4, num_classes=21, **kwargs):
    return get_voc_for_ssd(
        data_root=data_root,
        config=VGG_CONFIG(),
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
    )
