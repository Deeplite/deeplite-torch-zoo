from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.api.models.classification.imagenet.utils import (
    make_wrapper_func,
    load_checkpoint,
    NUM_IMAGENET_CLASSES,
)

from deeplite_torch_zoo.src.classification.imagenet_models.mobileone import mobileone
from deeplite_torch_zoo.src.classification.imagenet_models.ghostnet_v2 import (
    ghostnet_v2,
)
from deeplite_torch_zoo.src.classification.imagenet_models.edgevit import (
    edgevit_xxs,
    edgevit_xs,
    edgevit_s,
)
from torchvision.models import MobileNetV2

__all__ = [
    'mobilenetv2_w035',
    'get_edgevit_s',
    'get_edgevit_xs',
    'get_edgevit_xxs',
    'get_ghostnetv2_1',
    'get_ghostnetv2_13',
    'get_ghostnetv2_16',
]

MOBILEONE_BASE_URL = (
    'https://docs-assets.developer.apple.com/ml-research/datasets/mobileone'
)
GHOSTNETV2_BASE_URL = (
    'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2'
)

CHECKPOINT_URLS = {
    'mobilenetv2_w035_zoo': 'http://download.deeplite.ai/zoo/models/mobilenetv2_w035_imagenet_6020_4a56477132807d76.pt',
    'mobileone_s0_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s0_unfused.pth.tar',
    'mobileone_s1_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s1_unfused.pth.tar',
    'mobileone_s2_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s2_unfused.pth.tar',
    'mobileone_s3_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s3_unfused.pth.tar',
    'mobileone_s4_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s4_unfused.pth.tar',
    'ghostnetv2_1.0_zoo': f'{GHOSTNETV2_BASE_URL}/ck_ghostnetv2_10.pth.tar',
    'ghostnetv2_1.3_zoo': f'{GHOSTNETV2_BASE_URL}/ck_ghostnetv2_13.pth.tar',
    'ghostnetv2_1.6_zoo': f'{GHOSTNETV2_BASE_URL}/ck_ghostnetv2_16.pth.tar',
}


def get_mobileone(
    model_name, num_classes=NUM_IMAGENET_CLASSES, pretrained=False, device='cuda'
):
    model = mobileone(
        num_classes=num_classes, inference_mode=False, variant=model_name.split('_')[1]
    )
    if pretrained:
        model = load_checkpoint(model, model_name, 'imagenet', CHECKPOINT_URLS, device)
    return model.to(device)


for mobileone_variant in ('s0', 's1', 's2', 's3', 's4'):
    model_tag = f'mobileone_{mobileone_variant}_zoo'
    wrapper_tag = f'{model_tag}_imagenet'
    globals()[wrapper_tag] = make_wrapper_func(get_mobileone, wrapper_tag, model_tag)
    __all__.append(wrapper_tag)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenetv2_w035_zoo',
    dataset_name='imagenet',
    task_type='classification',
    has_checkpoint=True,
)
def mobilenetv2_w035(pretrained=False, device="cuda", num_classes=NUM_IMAGENET_CLASSES):
    model = MobileNetV2(width_mult=0.35, num_classes=num_classes)
    if pretrained:
        model = load_checkpoint(
            model, 'mobilenetv2_w035_zoo', 'imagenet', CHECKPOINT_URLS, device
        )
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='edgevit_xxs_zoo',
    dataset_name='imagenet',
    task_type='classification',
    has_checkpoint=False,
)
def get_edgevit_xxs(num_classes=NUM_IMAGENET_CLASSES, pretrained=False, device='cuda'):
    model = edgevit_xxs(num_classes=num_classes)
    if pretrained:
        model = load_checkpoint(
            model, 'edgevit_xxs_zoo', 'imagenet', CHECKPOINT_URLS, device
        )
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='edgevit_xs_zoo',
    dataset_name='imagenet',
    task_type='classification',
    has_checkpoint=False,
)
def get_edgevit_xs(num_classes=NUM_IMAGENET_CLASSES, pretrained=False, device='cuda'):
    model = edgevit_xs(num_classes=num_classes)
    if pretrained:
        model = load_checkpoint(
            model, 'edgevit_xs_zoo', 'imagenet', CHECKPOINT_URLS, device
        )
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='edgevit_s_zoo',
    dataset_name='imagenet',
    task_type='classification',
    has_checkpoint=False,
)
def get_edgevit_s(num_classes=NUM_IMAGENET_CLASSES, pretrained=False, device='cuda'):
    model = edgevit_s(num_classes=num_classes)
    if pretrained:
        model = load_checkpoint(
            model, 'edgevit_s_zoo', 'imagenet', CHECKPOINT_URLS, device
        )
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='ghostnetv2_1.0_zoo',
    dataset_name='imagenet',
    task_type='classification',
    has_checkpoint=False,
)
def get_ghostnetv2_1(num_classes=NUM_IMAGENET_CLASSES, pretrained=False, device='cuda'):
    model = ghostnet_v2(num_classes=num_classes, width=1.0)
    if pretrained:
        model = load_checkpoint(
            model, 'ghostnetv2_1.0_zoo', 'imagenet', CHECKPOINT_URLS, device
        )
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='ghostnetv2_1.3_zoo',
    dataset_name='imagenet',
    task_type='classification',
    has_checkpoint=False,
)
def get_ghostnetv2_13(
    num_classes=NUM_IMAGENET_CLASSES, pretrained=False, device='cuda'
):
    model = ghostnet_v2(num_classes=num_classes, width=1.3)
    if pretrained:
        model = load_checkpoint(
            model, 'ghostnetv2_1.3_zoo', 'imagenet', CHECKPOINT_URLS, device
        )
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='ghostnetv2_1.6_zoo',
    dataset_name='imagenet',
    task_type='classification',
    has_checkpoint=False,
)
def get_ghostnetv2_16(
    num_classes=NUM_IMAGENET_CLASSES, pretrained=False, device='cuda'
):
    model = ghostnet_v2(num_classes=num_classes, width=1.6)
    if pretrained:
        model = load_checkpoint(
            model, 'ghostnetv2_1.6_zoo', 'imagenet', CHECKPOINT_URLS, device
        )
    return model.to(device)
