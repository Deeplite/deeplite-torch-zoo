from deeplite_torch_zoo.wrappers.datasets.objectdetection.yolo import get_voc_for_yolo


__all__ = ["get_voc_for_ssd300"]


def get_voc_for_ssd300(
    data_root, batch_size=32, num_workers=4, num_classes=20, device="cuda", **kwargs
):
    return get_voc_for_yolo(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        num_classes=num_classes,
        img_size=300,
        device=device,
    )
