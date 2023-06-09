"""dense net in pytorch

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.src.classification.cifar_models.densenet import (
    Bottleneck,
    DenseNet,
)
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY

__all__ = [
    "densenet121_cifar100",
]

model_urls = {
    "densenet121": "http://download.deeplite.ai/zoo/models/densenet121-cifar100-7e4ec64b17b04532.pth",
}


def _densenet(
    arch,
    block,
    layers,
    growth_rate=32,
    pretrained=False,
    num_classes=100,
):
    model = DenseNet(block, layers, growth_rate, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)

    return model


@MODEL_WRAPPER_REGISTRY.register(
    model_name='densenet121',
    dataset_name='cifar100',
    task_type='classification'
)
def densenet121_cifar100(
    pretrained=False, num_classes=100
):
    return _densenet(
        "densenet121",
        Bottleneck,
        [6, 12, 24, 16],
        growth_rate=32,
        num_classes=num_classes,
        pretrained=pretrained,
    )


def densenet161(pretrained=False):
    return _densenet(
        "densenet161",
        Bottleneck,
        [6, 12, 36, 24],
        growth_rate=48,
        pretrained=pretrained,
    )


def densenet169(pretrained=False):
    return _densenet(
        "densenet169",
        Bottleneck,
        [6, 12, 32, 32],
        growth_rate=32,
        pretrained=pretrained,
    )


def densenet201(pretrained=False):
    return _densenet(
        "densenet201",
        Bottleneck,
        [6, 12, 48, 32],
        growth_rate=32,
        pretrained=pretrained,
    )
