from torch.hub import load_state_dict_from_url

__all__ = ["mlp2_mnist", "mlp4_mnist", "mlp8_mnist"]

from deeplite_torch_zoo.src.classification.mlp import MLP

model_urls = {
    "mlp2": "http://download.deeplite.ai/zoo/models/mlp2-mnist-cd7538f979ca4d0e.pth",
    "mlp4": "http://download.deeplite.ai/zoo/models/mlp4-mnist-c6614ff040df60a4.pth",
    "mlp8": "http://download.deeplite.ai/zoo/models/mlp8-mnist-de6f135822553043.pth",
}


def _mlp10_mnist(arch, n_hiddens, pretrained=False, progress=True, device="cuda"):
    model = MLP(input_dims=784, n_hiddens=n_hiddens, n_class=10)

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def mlp2_mnist(pretrained=False, progress=True, device="cuda"):
    return _mlp10_mnist(
        "mlp2",
        n_hiddens=[128, 128],
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def mlp4_mnist(pretrained=False, progress=True, device="cuda"):
    return _mlp10_mnist(
        "mlp4",
        n_hiddens=[128, 128, 128, 128],
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def mlp8_mnist(pretrained=False, progress=True, device="cuda"):
    return _mlp10_mnist(
        "mlp8",
        n_hiddens=[128, 128, 128, 128, 128, 128, 128, 128],
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
