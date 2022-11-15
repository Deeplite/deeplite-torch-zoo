from deeplite_torch_zoo.src.classification.mobilenets.mobilenetv1 import \
    MobileNetV1
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

__all__ = ["mobilenet_v1_vww", 'mobilenet_v1_025_vww', 'mobilenet_v1_025_96px_vww']

model_urls = {
    "mobilenet_v1": "http://download.deeplite.ai/zoo/models/mobilenetv1-vww-84f65dc4bc649cd6.pth",
    "mobilenet_v1_0.25": "http://download.deeplite.ai/zoo/models/mobilenet_v1_0.25.pt",
    "mobilenet_v1_0.25_96px": "http://download.deeplite.ai/zoo/models/mobilenet_v1_0.25_96px_798-8df2181bdab1433e.pt",
}

def mobilenetv1_vww(model_name, num_classes=2, last_pooling_size=7, width_mult=1., device='cuda', pretrained=False, progress=True):
    model = MobileNetV1(num_classes=num_classes, width_mult=width_mult, last_pooling_size=last_pooling_size)
    if pretrained:
        checkpoint_url = model_urls[model_name]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v1', dataset_name='vww', task_type='classification')
def mobilenet_v1_vww(pretrained=False, progress=True, num_classes=2, device='cuda'):
    return mobilenetv1_vww(model_name='mobilenet_v1', pretrained=pretrained, progress=progress,
        num_classes=num_classes, device=device)


@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v1_0.25', dataset_name='vww', task_type='classification')
def mobilenet_v1_025_vww(pretrained=False, progress=True, num_classes=2, device='cuda', width_mult=0.25):
    return mobilenetv1_vww(model_name='mobilenet_v1_0.25', pretrained=pretrained, progress=progress,
        num_classes=num_classes, device=device, width_mult=width_mult)


@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v1_0.25_96px', dataset_name='vww', task_type='classification')
def mobilenet_v1_025_96px_vww(pretrained=False, progress=True, num_classes=2, device='cuda',
    width_mult=0.25, last_pooling_size=3):
    return mobilenetv1_vww(model_name='mobilenet_v1_0.25_96px', pretrained=pretrained, progress=progress,
        num_classes=num_classes, device=device, width_mult=width_mult, last_pooling_size=last_pooling_size)
