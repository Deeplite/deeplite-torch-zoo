"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""


from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.classification.preact_resnet import PreActResNet, PreActBlock, PreActBottleneck

__all__ = [
    # 'pre_act_resnet34', 'pre_act_resnet50', 'pre_act_resnet101', 'pre_act_resnet152'
    "pre_act_resnet18_cifar100"
]

model_urls = {
    "pre_act_resnet18": "http://download.deeplite.ai/zoo/models/pre_act_resnet18-cifar100-1c4d1dc76ee9c6f6.pth",
    "pre_act_resnet34": "",
    "pre_act_resnet50": "",
    "pre_act_resnet101": "",
    "pre_act_resnet152": "",
}


def _pre_act_resnet(arch, block, layers, pretrained=False, progress=True, device='cuda'):
    model = PreActResNet(block, layers)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def pre_act_resnet18_cifar100(pretrained=False, progress=True, device='cuda'):
    return _pre_act_resnet(
        "pre_act_resnet18",
        PreActBlock,
        [2, 2, 2, 2],
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
