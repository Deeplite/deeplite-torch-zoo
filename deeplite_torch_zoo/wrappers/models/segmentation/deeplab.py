from deeplite_torch_zoo.src.segmentation.deeplab.repo.modeling.deeplab import DeepLab
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights


__all__ = ["deeplab_mobilenet_voc"]


model_urls = {
    "mobilenet_voc": "http://download.deeplite.ai/zoo/models/deeplab-mobilenet-voc-20_593-94ac51da679409d6.pth",
}


def deeplab(
    backbone="resnet",
    dataset="voc",
    num_classes=21,
    pretrained=False,
    progress=True,
    device="cuda",
):
    model = DeepLab(backbone=backbone, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[f"{backbone}_{dataset}"]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)

    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='deeplab_mobilenet', dataset_name='voc',
    task_type='semantic_segmentation')
def deeplab_mobilenet_voc(pretrained=True, progress=False, num_classes=21, device='cuda'):
    return deeplab(
        backbone="mobilenet",
        dataset="voc",
        num_classes=num_classes,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
