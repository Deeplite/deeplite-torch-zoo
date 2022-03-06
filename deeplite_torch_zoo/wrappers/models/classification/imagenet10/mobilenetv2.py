import torchvision
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = [
    "mobilenet_v2_0_35_imagenet10"
]

model_urls = {
    "mobilenetv2_0.35": "http://download.deeplite.ai/zoo/models/mobilenetv2_0.35-imagenet10-2410796e32dbde1c.pth",
    "mobilenetv2_1.0": "",
}


def _mobilenetv2_imagenet10(arch, alpha=1.0, pretrained=False, progress=True, num_classes=10, device='cuda'):
    model = torchvision.models.mobilenet.MobileNetV2(width_mult=alpha, num_classes=num_classes)

    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v2_0_35', dataset_name='imagenet10', task_type='classification')
def mobilenet_v2_0_35_imagenet10(pretrained=False, progress=True, num_classes=10, device='cuda'):
    return _mobilenetv2_imagenet10(
        "mobilenetv2_0.35", alpha=0.35, pretrained=pretrained, progress=progress, num_classes=num_classes, device=device
    )


def mobilenet_v2_1_0(pretrained=False, progress=True, device='cuda'):
    return _mobilenetv2_imagenet10(
        "mobilenetv2_1.0", alpha=1, pretrained=pretrained, progress=progress, device=device
    )
