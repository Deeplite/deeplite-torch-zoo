"""resnext in pytorch

[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He.
    Aggregated Residual Transformations for Deep Neural Networks
    https://arxiv.org/abs/1611.05431
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.resnext import ResNeXt
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = [
    "resnext29_2x64d_cifar100"
]

model_urls = {
    "resnext29_2x64d": "http://download.deeplite.ai/zoo/models/resnext29_2x64d-cifar100-f6ba33baf30048d1.pth",
}


def _resnext(
    arch, num_blocks, cardinality, bottleneck_width, num_classes=100, pretrained=False, progress=True, device='cuda'
):
    model = ResNeXt(
        num_blocks=num_blocks,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        num_classes=num_classes,
    )
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='resnext29_2x64d', dataset_name='cifar100', task_type='classification')
def resnext29_2x64d_cifar100(pretrained=False, progress=True, num_classes=100, device='cuda'):
    return _resnext(
        "resnext29_2x64d",
        num_classes=num_classes,
        num_blocks=[3, 3, 3],
        cardinality=2,
        bottleneck_width=64,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def resnext29_4x64d(pretrained=False, progress=True, device='cuda'):
    return _resnext(
        "resnext29_4x64d",
        num_blocks=[3, 3, 3],
        cardinality=4,
        bottleneck_width=64,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def resnext29_8x64d(pretrained=False, progress=True, device='cuda'):
    return _resnext(
        "resnext29_8x64d",
        num_blocks=[3, 3, 3],
        cardinality=8,
        bottleneck_width=64,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def resnext29_32x4d(pretrained=False, progress=True, device='cuda'):
    return _resnext(
        "resnext29_32x4d",
        num_blocks=[3, 3, 3],
        cardinality=32,
        bottleneck_width=4,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
