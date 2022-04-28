
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["fasterrcnn_resnet50_fpn_coco"]


@MODEL_WRAPPER_REGISTRY.register(model_name='fasterrcnn_resnet50_fpn', dataset_name='coco',
    task_type='object_detection')
def fasterrcnn_resnet50_fpn_coco(pretrained=False, progress=True, device="cuda"):
    return fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress)
