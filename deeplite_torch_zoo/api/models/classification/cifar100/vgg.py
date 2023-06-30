"""vgg in pytorch

[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.vgg import VGG
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["vgg19_cifar100"]

model_urls = {
    "vgg19": "http://download.deeplite.ai/zoo/models/vgg19-cifar100-6d791de492a133b6.pth",
}


def _vgg(cfg, pretrained=False, num_classes=100):
    model = VGG(cfg, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[cfg]
        model = load_pretrained_weights(model, checkpoint_url)
    return model


def vgg11(pretrained=False):
    return _vgg("vgg11", pretrained)


def vgg13(pretrained=False):
    return _vgg("vgg13", pretrained)


def vgg16(pretrained=False):
    return _vgg("vgg16", pretrained)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='vgg19', dataset_name='cifar100', task_type='classification'
)
def vgg19_cifar100(pretrained=False, num_classes=100):
    return _vgg("vgg19", pretrained, num_classes=num_classes)
