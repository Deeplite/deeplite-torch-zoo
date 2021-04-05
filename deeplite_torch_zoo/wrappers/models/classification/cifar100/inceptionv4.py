""" inceptionv4 in pytorch

[1] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    https://arxiv.org/abs/1602.07261
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.classification.inceptionv4 import InceptionResNetV2, InceptionV4


__all__ = [
    # 'inception_resnet_v2'
    "inception_v4_cifar100"
]

model_urls = {
    "inception_v4": "http://download.deeplite.ai/zoo/models/inceptionv4-cifar100-ad655dfc5fe5b02f.pth",
    "inception_resnet_v2": "",
}


def _inceptionv4(arch, A, B, C, pretrained=False, progress=True, device='cuda'):
    model = InceptionV4(A, B, C)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def _inception_resnet_v2(arch, A, B, C, pretrained=False, progress=True):
    model = InceptionResNetV2(A, B, C)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model


def inception_v4_cifar100(pretrained=False, progress=True, device='cuda'):
    return _inceptionv4(
        "inception_v4", 4, 7, 3, pretrained=pretrained, progress=progress, device=device
    )


def inception_resnet_v2(pretrained=False, progress=True):
    return _inception_resnet_v2(
        "inception_resnet_v2", 5, 10, 5, pretrained=pretrained, progress=progress
    )
