from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.api.models.classification.imagenet.utils import (
    load_checkpoint,
    NUM_IMAGENET_CLASSES,
)

from deeplite_torch_zoo.src.classification.imagenet_models.mobileone import mobileone
from deeplite_torch_zoo.src.classification.imagenet_models.ghostnetv2 import (
    ghostnet_v2,
)
from deeplite_torch_zoo.src.classification.imagenet_models.edgevit import (
    edgevit_xxs,
    edgevit_xs,
    edgevit_s,
)
from deeplite_torch_zoo.src.classification.imagenet_models.fasternet import (
    fasternet_t0,
    fasternet_t1,
    fasternet_t2,
    fasternet_s,
    fasternet_m,
    fasternet_l,
)
from torchvision.models import MobileNetV2


__all__ = []

MOBILEONE_BASE_URL = (
    'https://docs-assets.developer.apple.com/ml-research/datasets/mobileone'
)


CHECKPOINT_URLS = {
    'mobilenetv2_w035_zoo_imagenet': 'http://download.deeplite.ai/zoo/models/mobilenetv2_w035_imagenet_6020_4a56477132807d76.pt',
    'mobileone_s0_zoo_imagenet': f'{MOBILEONE_BASE_URL}/mobileone_s0_unfused.pth.tar',
    'mobileone_s1_zoo_imagenet': f'{MOBILEONE_BASE_URL}/mobileone_s1_unfused.pth.tar',
    'mobileone_s2_zoo_imagenet': f'{MOBILEONE_BASE_URL}/mobileone_s2_unfused.pth.tar',
    'mobileone_s3_zoo_imagenet': f'{MOBILEONE_BASE_URL}/mobileone_s3_unfused.pth.tar',
    'mobileone_s4_zoo_imagenet': f'{MOBILEONE_BASE_URL}/mobileone_s4_unfused.pth.tar',
}


MODEL_FNS = {
    'edgevit_s': (edgevit_s, {}),
    'edgevit_xs': (edgevit_xs, {}),
    'edgevit_xxs': (edgevit_xxs, {}),
    'fasternet_t0': (fasternet_t0, {}),
    'fasternet_t1': (fasternet_t1, {}),
    'fasternet_t2': (fasternet_t2, {}),
    'fasternet_s': (fasternet_s, {}),
    'fasternet_m': (fasternet_m, {}),
    'fasternet_l': (fasternet_l, {}),
    'mobilenetv2_w035': (MobileNetV2, {'width_mult': 0.35}),
    'mobileone_s0': (mobileone, {'inference_mode': False, 'variant': 's0'}),
    'mobileone_s1': (mobileone, {'inference_mode': False, 'variant': 's1'}),
    'mobileone_s2': (mobileone, {'inference_mode': False, 'variant': 's2'}),
    'mobileone_s3': (mobileone, {'inference_mode': False, 'variant': 's3'}),
    'mobileone_s4': (mobileone, {'inference_mode': False, 'variant': 's4'}),
    'ghostnetv2_1.0': (ghostnet_v2, {'width': 1.0}),
    'ghostnetv2_1.3': (ghostnet_v2, {'width': 1.3}),
    'ghostnetv2_1.6': (ghostnet_v2, {'width': 1.6}),
}


def register_model_wrapper(model_fn, model_name, **model_kwargs):

    def get_model(num_classes=NUM_IMAGENET_CLASSES, pretrained=False):
        model = model_fn(num_classes=num_classes, **model_kwargs)
        if pretrained:
            model = load_checkpoint(
                model, model_name, 'imagenet', CHECKPOINT_URLS
            )
        return model

    get_model = MODEL_WRAPPER_REGISTRY.register(
        model_name=model_name,
        dataset_name='imagenet',
        task_type='classification',
        has_checkpoint=f'{model_name}_imagenet' in CHECKPOINT_URLS,
    )(get_model)

    get_model.__name__ = f'{model_name}_imagenet'
    return get_model


for _model_name, (_model_fn, _model_kwargs) in MODEL_FNS.items():
    wrapper_fn = register_model_wrapper(_model_fn, f'{_model_name}_zoo', **_model_kwargs)
    globals()[wrapper_fn.__name__] = wrapper_fn
    __all__.append(wrapper_fn.__name__)
