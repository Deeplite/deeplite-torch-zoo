"""dense net in pytorch

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.classification.densenet import DenseNet, Bottleneck


__all__ = [
    # 'densenet161', 'densenet169', 'densenet201'
    "densenet121_cifar100"
]

model_urls = {
    "densenet121": "http://download.deeplite.ai/zoo/models/densenet121-cifar100-7e4ec64b17b04532.pth",
    "densenet161": "",
    "densenet169": "",
    "densenet201": "",
}


def _densenet(arch, block, layers, growth_rate=32, pretrained=False, progress=True, device='cuda'):
    model = DenseNet(block, layers, growth_rate)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def densenet121_cifar100(pretrained=False, progress=True, device='cuda'):
    return _densenet(
        "densenet121",
        Bottleneck,
        [6, 12, 24, 16],
        growth_rate=32,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def densenet161(pretrained=False, progress=True):
    return _densenet(
        "densenet161",
        Bottleneck,
        [6, 12, 36, 24],
        growth_rate=48,
        pretrained=pretrained,
        progress=progress,
    )


def densenet169(pretrained=False, progress=True):
    return _densenet(
        "densenet169",
        Bottleneck,
        [6, 12, 32, 32],
        growth_rate=32,
        pretrained=pretrained,
        progress=progress,
    )


def densenet201(pretrained=False, progress=True):
    return _densenet(
        "densenet201",
        Bottleneck,
        [6, 12, 48, 32],
        growth_rate=32,
        pretrained=pretrained,
        progress=progress,
    )
