from torchvision import models
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

__all__ = ["vgg19_tinyimagenet"]

model_urls = {
    "vgg19": "http://download.deeplite.ai/zoo/models/vgg19_tinyimagenet_0-7288-aaa20280ea9bb886.pt",
}


@MODEL_WRAPPER_REGISTRY.register(model_name='vgg19', dataset_name='tinyimagenet', task_type='classification')
def vgg19_tinyimagenet(pretrained=False, progress=True, num_classes=100, device="cuda"):
    model = models.vgg19(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls['vgg19']
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)
