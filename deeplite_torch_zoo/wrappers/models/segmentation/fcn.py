from torchfcn.models import FCN32s as FCN
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights


__all__ = ["fcn32_voc_20"]


model_urls = {
    "fcn32_voc_20": "http://download.deeplite.ai/zoo/models/fcn32-voc-20_713-b745bd7e373e31d1.pth",
}


def fcn32(
    net="fcn32",
    dataset="voc_20",
    num_classes=21,
    pretrained=False,
    progress=True,
    device="cuda",
):
    model = FCN(n_class=num_classes)
    if pretrained:
        checkpoint_url = model_urls[f"{net}_{dataset}"]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)

    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='fcn32', dataset_name='voc_20', task_type='semantic_segmentation')
def fcn32_voc_20(pretrained=True, progress=False, num_classes=21, device='cuda'):
    return fcn32(
        net="fcn32",
        dataset="voc_20",
        num_classes=num_classes,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
