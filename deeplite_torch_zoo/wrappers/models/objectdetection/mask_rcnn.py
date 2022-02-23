
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["fasterrcnn_resnet50_fpn_coco_80"]


@MODEL_WRAPPER_REGISTRY.register(model_name='fasterrcnn_resnet50_fpn', dataset_name='coco_80',
    task_type='classification')
def fasterrcnn_resnet50_fpn_coco_80(pretrained=False, progress=True, device="cuda"):
    return fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress)
