from deeplite_torch_zoo.src.classification.imagenet_models.mobileone import \
    mobileone
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY

__all__ = []

BASE_URL = 'https://docs-assets.developer.apple.com/ml-research/datasets/mobileone'
MODEL_URLS = {
    'mobileone_s0': f'{BASE_URL}/mobileone_s0_unfused.pth.tar',
    'mobileone_s1': f'{BASE_URL}/mobileone_s1_unfused.pth.tar',
    'mobileone_s2': f'{BASE_URL}/mobileone_s2_unfused.pth.tar',
    'mobileone_s3': f'{BASE_URL}/mobileone_s3_unfused.pth.tar',
    'mobileone_s4': f'{BASE_URL}/mobileone_s4_unfused.pth.tar',
}
MODEL_VARIANTS = ('s0', 's1', 's2', 's3', 's4')


def get_mobileone(model_name, num_classes=1000, pretrained=False, progress=True, device='cuda'):

    model = mobileone(
        num_classes=num_classes,
        inference_mode=False,
        variant=model_name.split('_')[1]
    )

    if pretrained:
        checkpoint_url = MODEL_URLS[model_name]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)

    return model.to(device)


def make_wrapper_func(wrapper_name, model_name, dataset_name='imagenet', num_classes=1000):
    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset_name,
        task_type='classification', has_checkpoint=True)
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device='cuda'):
        return get_mobileone(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            progress=progress,
            device=device,
        )
    wrapper_func.__name__ = wrapper_name
    return wrapper_func


for mobileone_variant in MODEL_VARIANTS:
    model_tag = f'mobileone_{mobileone_variant}'
    wrapper_tag = f'{model_tag}_imagenet'
    globals()[wrapper_tag] = make_wrapper_func(wrapper_tag, model_tag)
    __all__.append(wrapper_tag)
