from deeplite_torch_zoo.src.classification.imagenet_models.mobileone import mobileone
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.api.models.classification.imagenet.utils import make_wrapper_func, NUM_IMAGENET_CLASSES

from torchvision.models import MobileNetV2

__all__ = [
    'mobilenetv2_w035',
]

MOBILEONE_BASE_URL = 'https://docs-assets.developer.apple.com/ml-research/datasets/mobileone'
CHECKPOINT_URLS = {
    'mobilenetv2_w035': 'http://download.deeplite.ai/zoo/models/mobilenetv2_w035_imagenet_6020_4a56477132807d76.pt',
    'mobileone_s0_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s0_unfused.pth.tar',
    'mobileone_s1_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s1_unfused.pth.tar',
    'mobileone_s2_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s2_unfused.pth.tar',
    'mobileone_s3_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s3_unfused.pth.tar',
    'mobileone_s4_zoo': f'{MOBILEONE_BASE_URL}/mobileone_s4_unfused.pth.tar',
}


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenetv2_w035_zoo',
    dataset_name='imagenet',
    task_type='classification',
    has_checkpoint=True,
)
def mobilenetv2_w035(pretrained=False, device="cuda", num_classes=NUM_IMAGENET_CLASSES):
    model = MobileNetV2(width_mult=0.35, num_classes=num_classes)

    if pretrained:
        checkpoint_url = CHECKPOINT_URLS['mobilenetv2_w035']
        model = load_pretrained_weights(model, checkpoint_url, device)

    return model.to(device)


def get_mobileone(
    model_name, num_classes=NUM_IMAGENET_CLASSES, pretrained=False, device='cuda'
):
    model = mobileone(
        num_classes=num_classes, inference_mode=False, variant=model_name.split('_')[1]
    )

    if pretrained:
        checkpoint_url = CHECKPOINT_URLS[model_name]
        model = load_pretrained_weights(model, checkpoint_url, device)

    return model.to(device)


for mobileone_variant in ('s0', 's1', 's2', 's3', 's4'):
    model_tag = f'mobileone_{mobileone_variant}_zoo'
    wrapper_tag = f'{model_tag}_imagenet'
    globals()[wrapper_tag] = make_wrapper_func(get_mobileone, wrapper_tag, model_tag)
    __all__.append(wrapper_tag)
