"""mobilenetv2 in pytorch

[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""


from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.classification.mobilenetv2 import MobileNetV2


__all__ = ["mobilenet_v2_cifar100"]

model_urls = {
    "mobilenet_v2": "http://download.deeplite.ai/zoo/models/mobilenetv2-cifar100-a7ba34049d626cf4.pth",
}




def _mobilenetv2(arch, pretrained=False, progress=True, device='cuda'):
    model = MobileNetV2()
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def mobilenet_v2_cifar100(pretrained=False, progress=True, device='cuda'):
    return _mobilenetv2("mobilenet_v2", pretrained, progress, device=device)
