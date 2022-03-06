"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""


from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.preact_resnet import PreActResNet, PreActBlock, PreActBottleneck
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = [
    "pre_act_resnet18_cifar100"
]

model_urls = {
    "pre_act_resnet18": "http://download.deeplite.ai/zoo/models/pre_act_resnet18-cifar100-1c4d1dc76ee9c6f6.pth",
}


def _pre_act_resnet(arch, block, layers,  num_classes=100, pretrained=False, progress=True, device='cuda'):
    model = PreActResNet(block, layers, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='pre_act_resnet18', dataset_name='cifar100', task_type='classification')
def pre_act_resnet18_cifar100(pretrained=False, progress=True, num_classes=100, device='cuda'):
    return _pre_act_resnet(
        "pre_act_resnet18",
        PreActBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def pre_act_resnet34(pretrained=False, progress=True):
    return _pre_act_resnet(
        "pre_act_resnet34",
        PreActBlock,
        [3, 4, 6, 3],
        pretrained=pretrained,
        progress=progress,
    )


def pre_act_resnet50(pretrained=False, progress=True):
    return _pre_act_resnet(
        "pre_act_resnet50",
        PreActBottleneck,
        [3, 4, 6, 3],
        pretrained=pretrained,
        progress=progress,
    )


def pre_act_resnet101(pretrained=False, progress=True):
    return _pre_act_resnet(
        "pre_act_resnet101",
        PreActBottleneck,
        [3, 4, 23, 3],
        pretrained=pretrained,
        progress=progress,
    )


def pre_act_resnet152(pretrained=False, progress=True):
    return _pre_act_resnet(
        "pre_act_resnet152",
        PreActBottleneck,
        [3, 8, 36, 3],
        pretrained=pretrained,
        progress=progress,
    )
