from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.mobilenets.mobilenetv1 import MobileNetV1
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mobilenet_v1_vww"]

model_urls = {
    "mobilenetv1": "http://download.deeplite.ai/zoo/models/mobilenetv1-vww-84f65dc4bc649cd6.pth",
    "mobilenetv1_0.25": "http://download.deeplite.ai/zoo/models/mobilenet_v1_0.25.pt"
}


def _mobilenetv1_vww(arch, pretrained=False, progress=True, num_classes=2, device='cuda', width_mult=1.):
    model = MobileNetV1(num_classes=num_classes, width_mult=width_mult)

    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v1', dataset_name='vww', task_type='classification')
def mobilenet_v1_vww(pretrained=False, progress=True, num_classes=2, device='cuda'):
    return _mobilenetv1_vww("mobilenetv1", pretrained=pretrained, progress=progress, num_classes=num_classes, device=device)

@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v1_0.25', dataset_name='vww', task_type='classification')
def mobilenet_v1_025_vww(pretrained=False, progress=True, num_classes=2, device='cuda',width_mult=0.25):
    return _mobilenetv1_vww("mobilenetv1_0.25", pretrained=pretrained, progress=progress, num_classes=num_classes, device=device, width_mult=width_mult)
