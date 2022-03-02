"""mobilenet in pytorch

[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.mobilenetv1 import MobileNet
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mobilenet_v1_cifar100"]

model_urls = {
    "mobilenet_v1": "http://download.deeplite.ai/zoo/models/mobilenetv1-cifar100-4690c1a2246529eb.pth",
}




def _mobilenetv1(arch, pretrained=False, progress=True, num_classes=100, device='cuda'):
    model = MobileNet(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v1', dataset_name='cifar100', task_type='classification')
def mobilenet_v1_cifar100(pretrained=False, num_classes=100, progress=True, device='cuda'):
    return _mobilenetv1("mobilenet_v1", pretrained, progress, num_classes=num_classes, device=device)
