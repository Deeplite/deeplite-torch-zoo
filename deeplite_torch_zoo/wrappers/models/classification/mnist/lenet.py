"""

[1] Implementation
    https://github.com/kuangliu/pytorch-cifar

"""


from torch.hub import load_state_dict_from_url
from deeplite_torch_zoo.src.classification.lenet import LeNet5


__all__ = ["lenet5_mnist"]

model_urls = {
    "lenet5": "http://download.deeplite.ai/zoo/models/lenet-mnist-e5e2d99e08460491.pth",
}




def _lenet_mnist(arch, pretrained=False, progress=True, device="cuda"):
    model = LeNet5()

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def lenet5_mnist(pretrained=False, progress=True, device="cuda"):
    return _lenet_mnist(
        "lenet5", pretrained=pretrained, progress=progress, device=device
    )
