"""mobilenet in pytorch

[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.classification.mobilenetv1 import MobileNet


__all__ = ["mobilenet_v1_cifar100"]

model_urls = {
    "mobilenet_v1": "http://download.deeplite.ai/zoo/models/mobilenetv1-cifar100-4690c1a2246529eb.pth",
}




def _mobilenetv1(arch, pretrained=False, progress=True, device='cuda'):
    model = MobileNet()
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def mobilenet_v1_cifar100(pretrained=False, progress=True, device='cuda'):
    return _mobilenetv1("mobilenet_v1", pretrained, progress, device=device)
