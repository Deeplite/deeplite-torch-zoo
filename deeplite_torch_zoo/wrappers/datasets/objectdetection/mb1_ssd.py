from deeplite_torch_zoo.src.objectdetection.ssd.config.mobilenetv1_ssd_config import (
    MOBILENET_CONFIG,
)
from deeplite_torch_zoo.wrappers.datasets.objectdetection.ssd import get_voc_for_ssd


__all__ = ["get_voc_for_mb1_ssd"]


def get_voc_for_mb1_ssd(data_root, batch_size=32, num_workers=4, num_classes=21,
						fp16=False, distributed=False, device="cuda", **kwargs):
    return get_voc_for_ssd(
        data_root=data_root,
        config=MOBILENET_CONFIG(),
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        fp16=fp16,
        distributed=distributed,
        device=device
    )
