
from torchvision import models
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mobilenet_v2_tinyimagenet"]


model_urls = {
    "mobilenet_v2": "http://download.deeplite.ai/zoo/models/mobilenet_v2_tinyimagenet_0-6803-4ec21929b72f0b4d.pt",
}


@MODEL_WRAPPER_REGISTRY.register('mobilenet_v2', 'tinyimagenet')
def mobilenet_v2_tinyimagenet(pretrained=False, progress=True, num_classes=100, device="cuda"):
    model = models.mobilenet_v2(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls['mobilenet_v2']
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)
