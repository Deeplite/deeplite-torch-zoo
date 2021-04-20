
from deeplite_torch_zoo.src.poseestimation.datasets.datasets.utils import get_data_loaders


__all__ = ["get_coco_for_keypointrcnn_resnet50_fpn"]


def get_coco_for_keypointrcnn_resnet50_fpn(**kwargs):
	return get_data_loaders(dataset_type="coco_384x288")