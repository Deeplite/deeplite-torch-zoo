from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from torchvision.models import MobileNetV2

__all__ = ["mobilenetv2_w035", ]

model_urls = {
    "mobilenetv2_w035": "http://download.deeplite.ai/zoo/models/mobilenetv2_w035_imagenet_6020_4a56477132807d76.pt",
}


@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenetv2_w035', dataset_name='imagenet', task_type='classification')
def mobilenetv2_w035(pretrained=False, progress=True, device="cuda", num_classes=1000):
    model = MobileNetV2(width_mult=0.35, num_classes=num_classes)

    if pretrained:
        checkpoint_url = model_urls['mobilenetv2_w035']
        model = load_pretrained_weights(model, checkpoint_url, progress, device)

    return model.to(device)
