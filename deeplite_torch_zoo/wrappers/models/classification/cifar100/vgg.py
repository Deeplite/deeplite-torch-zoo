"""vgg in pytorch

[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.classification.vgg import VGG


__all__ = [
    # 'vgg11', 'vgg13', 'vgg16'
    "vgg19_cifar100"
]

model_urls = {
    "vgg11": "",
    "vgg13": "",
    "vgg16": "",
    "vgg19": "http://download.deeplite.ai/zoo/models/vgg19-cifar100-6d791de492a133b6.pth",
}


def _vgg(cfg, pretrained=False, progress=True, device='cuda'):
    model = VGG(cfg)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[cfg], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def vgg11(pretrained=False, progress=True, device='cuda'):
    return _vgg("vgg11", pretrained, progress, device=device)


def vgg13(pretrained=False, progress=True, device='cuda'):
    return _vgg("vgg13", pretrained, progress, device=device)


def vgg16(pretrained=False, progress=True, device='cuda'):
    return _vgg("vgg16", pretrained, progress, device=device)


def vgg19_cifar100(pretrained=False, progress=True, device='cuda'):
    return _vgg("vgg19", pretrained, progress, device=device)
