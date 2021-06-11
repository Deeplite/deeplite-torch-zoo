import sys
from deeplite_torch_zoo.wrappers.datasets.objectdetection.yolo import get_voc_for_yolo


__all__ = ["get_voc_for_ssd300"]


def get_voc_for_ssd300(
    data_root, batch_size=32, num_workers=4, num_classes=20, fp16=False, distributed=False, device="cuda", **kwargs
):
    if len(kwargs):
        print(f"Warning, {sys._getframe().f_code.co_name}: extra arguments {list(kwargs.keys())}!")

    return get_voc_for_yolo(
        data_root=data_root, batch_size=batch_size, num_workers=num_workers, num_classes=num_classes,
        img_size=300, fp16=fp16, distributed=distributed, device=device
    )
