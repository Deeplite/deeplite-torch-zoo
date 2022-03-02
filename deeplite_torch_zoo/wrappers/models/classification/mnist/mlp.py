from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mlp2_mnist", "mlp4_mnist", "mlp8_mnist"]

from deeplite_torch_zoo.src.classification.mnist_models.mlp import MLP

model_urls = {
    "mlp2": "http://download.deeplite.ai/zoo/models/mlp2-mnist-cd7538f979ca4d0e.pth",
    "mlp4": "http://download.deeplite.ai/zoo/models/mlp4-mnist-c6614ff040df60a4.pth",
    "mlp8": "http://download.deeplite.ai/zoo/models/mlp8-mnist-de6f135822553043.pth",
}


def _mlp10_mnist(arch, n_hiddens, pretrained=False, progress=True, num_classes=10, device="cuda"):
    model = MLP(input_dims=784, n_hiddens=n_hiddens, n_class=num_classes)

    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='mlp2', dataset_name='mnist', task_type='classification')
def mlp2_mnist(pretrained=False, progress=True, num_classes=10, device="cuda"):
    return _mlp10_mnist(
        "mlp2",
        n_hiddens=[128, 128],
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        device=device,
    )


@MODEL_WRAPPER_REGISTRY.register(model_name='mlp4', dataset_name='mnist', task_type='classification')
def mlp4_mnist(pretrained=False, progress=True, num_classes=10, device="cuda"):
    return _mlp10_mnist(
        "mlp4",
        n_hiddens=[128, 128, 128, 128],
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        device=device,
    )


@MODEL_WRAPPER_REGISTRY.register(model_name='mlp8', dataset_name='mnist', task_type='classification')
def mlp8_mnist(pretrained=False, progress=True, num_classes=10, device="cuda"):
    return _mlp10_mnist(
        "mlp8",
        n_hiddens=[128, 128, 128, 128, 128, 128, 128, 128],
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        device=device,
    )
