"""google net in pytorch

[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.googlenet import GoogLeNet
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

__all__ = ["googlenet_cifar100"]

model_urls = {
    "googlenet": "http://download.deeplite.ai/zoo/models/googlenet-cifar100-15f970a22f56433f.pth",
}



def _googlenet(arch, pretrained=False, progress=True, num_classes=100, device='cuda'):
    model = GoogLeNet(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='googlenet', dataset_name='cifar100', task_type='classification')
def googlenet_cifar100(pretrained=False, num_classes=100, progress=True, device='cuda'):
    return _googlenet("googlenet", pretrained, progress, device=device, num_classes=num_classes)
