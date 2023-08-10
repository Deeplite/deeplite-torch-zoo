"""

[1] Implementation
    https://github.com/kuangliu/pytorch-cifar

"""

from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.mnist_models.lenet import LeNet5
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["lenet5_mnist"]

model_urls = {
    "lenet5": "http://download.deeplite.ai/zoo/models/lenet-mnist-e5e2d99e08460491.pth",
}


def _lenet_mnist(arch, pretrained=False, num_classes=10):
    model = LeNet5(output=num_classes)

    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model


@MODEL_WRAPPER_REGISTRY.register(
    model_name='lenet5', dataset_name='mnist', task_type='classification'
)
def lenet5_mnist(pretrained=False, num_classes=10):
    return _lenet_mnist(
        "lenet5",
        pretrained=pretrained,
        num_classes=num_classes,
    )
