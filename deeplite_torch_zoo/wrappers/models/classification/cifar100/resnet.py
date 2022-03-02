"""resnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.resnet import ResNet, BasicBlock, Bottleneck
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = [
    "resnet18_cifar100",
    "resnet50_cifar100",
]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-cifar100-86b0c368c511bd57.pth",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50-cifar100-d03f14e3031410de.pth",
}



def _resnet(arch, block, layers, num_classes=100, pretrained=False, progress=True, device='cuda'):
    model = ResNet(block, layers, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet18', dataset_name='cifar100', task_type='classification')
def resnet18_cifar100(pretrained=False, progress=True, num_classes=100, device='cuda'):
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], num_classes=num_classes, pretrained=pretrained, progress=progress, device=device
    )


def resnet34(pretrained=False, progress=True, device='cuda'):
    return _resnet(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained=pretrained, progress=progress, device=device
    )


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet50', dataset_name='cifar100', task_type='classification')
def resnet50_cifar100(pretrained=False, progress=True, num_classes=100, device='cuda'):
    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], num_classes=num_classes, pretrained=pretrained, progress=progress, device=device
    )


def resnet101(pretrained=False, progress=True, device='cuda'):
    return _resnet(
        "resnet34", Bottleneck, [3, 4, 23, 3], pretrained=pretrained, progress=progress, device=device
    )


def resnet152(pretrained=False, progress=True, device='cuda'):
    return _resnet(
        "resnet34", Bottleneck, [3, 8, 36, 3], pretrained=pretrained, progress=progress, device=device
    )
