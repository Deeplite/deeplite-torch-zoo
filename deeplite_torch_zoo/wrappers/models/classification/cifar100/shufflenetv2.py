"""shufflenetv2 in pytorch

[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""


from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.shufflenetv2 import ShuffleNetV2
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["shufflenet_v2_1_0_cifar100"]


model_urls = {
    "shufflenet_v2": "http://download.deeplite.ai/zoo/models/shufflenet_v2_l.0-cifar100-16ae6f50f5adecad.pth",
}


def _shufflenetv2(arch, net_size=1, pretrained=False, num_classes=100, progress=True, device='cuda'):
    model = ShuffleNetV2(net_size, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='shufflenet_v2_1_0', dataset_name='cifar100', task_type='classification')
def shufflenet_v2_1_0_cifar100(pretrained=False, progress=True, num_classes=100, device='cuda'):
    return _shufflenetv2(
        "shufflenet_v2", net_size=1, num_classes=num_classes, pretrained=pretrained, progress=progress, device=device
    )
