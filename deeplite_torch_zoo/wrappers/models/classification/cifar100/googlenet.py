"""google net in pytorch

[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.classification.googlenet import GoogLeNet


__all__ = ["googlenet_cifar100"]

model_urls = {
    "googlenet": "http://download.deeplite.ai/zoo/models/googlenet-cifar100-15f970a22f56433f.pth",
}



def _googlenet(arch, pretrained=False, progress=True, device='cuda'):
    model = GoogLeNet()
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=False
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def googlenet_cifar100(pretrained=False, progress=True, device='cuda'):
    return _googlenet("googlenet", pretrained, progress, device=device)
