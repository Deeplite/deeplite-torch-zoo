from deeplite_torch_zoo.wrappers.datasets.objectdetection.mb1_ssd import get_voc_for_mb1_ssd


__all__ = ["get_voc_for_mb2_ssd_lite"]


def get_voc_for_mb2_ssd_lite(data_root, batch_size=32, num_workers=4, num_classes=21, **kwargs):
    return get_voc_for_mb1_ssd(
        data_root=data_root, num_classes=num_classes, batch_size=batch_size, num_workers=num_workers
    )
