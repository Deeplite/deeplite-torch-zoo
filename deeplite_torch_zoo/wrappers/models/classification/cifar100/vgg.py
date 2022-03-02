"""vgg in pytorch

[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.vgg import VGG
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = [
    "vgg19_cifar100"
]

model_urls = {
    "vgg19": "http://download.deeplite.ai/zoo/models/vgg19-cifar100-6d791de492a133b6.pth",
}


def _vgg(cfg, pretrained=False, progress=True, num_classes=100, device='cuda'):
    model = VGG(cfg, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[cfg]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def vgg11(pretrained=False, progress=True, device='cuda'):
    return _vgg("vgg11", pretrained, progress, device=device)


def vgg13(pretrained=False, progress=True, device='cuda'):
    return _vgg("vgg13", pretrained, progress, device=device)


def vgg16(pretrained=False, progress=True, device='cuda'):
    return _vgg("vgg16", pretrained, progress, device=device)


@MODEL_WRAPPER_REGISTRY.register(model_name='vgg19', dataset_name='cifar100', task_type='classification')
def vgg19_cifar100(pretrained=False, progress=True, num_classes=100, device='cuda'):
    return _vgg("vgg19", pretrained, progress, num_classes=num_classes, device=device)
