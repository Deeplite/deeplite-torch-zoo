import urllib.parse as urlparse
from functools import partial
from collections import namedtuple

from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY

from deeplite_torch_zoo.src.objectdetection.ssd.models.resnet_ssd import create_resnet_ssd
from deeplite_torch_zoo.src.objectdetection.ssd.models.mobilenetv2_ssd import create_mobilenetv2_ssd
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.vgg_ssd import create_vgg_ssd
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

from deeplite_torch_zoo.src.objectdetection.ssd.config.vgg_ssd_config import VGG_CONFIG
from deeplite_torch_zoo.src.objectdetection.ssd.config.mobilenetv1_ssd_config import MOBILENET_CONFIG



__all__ = []


CHECKPOINT_STORAGE_URL = "http://download.deeplite.ai/zoo/models/"

model_urls = {
    "resnet18_ssd_voc": "resnet18-ssd_voc_AP_0_728-564518d0c865972b.pth",
    "resnet34_ssd_voc": "resnet34_ssd-voc_760-a102a7ca6564ab44.pth",
    "resnet50_ssd_voc": "resnet50_ssd-voc_766-d934cbe063398fcd.pth",
    "mb2_ssd_coco": "mb2-ssd-coco-80-0_303-78c63d5b07edaeaf.pth",
    "mb2_ssd_voc": "mb2-ssd-voc-20-0_440-8402e1cfc3e076b1.pth",
    "vgg16_ssd_voc": "vgg16-ssd-voc-mp-0_7726-b1264e8beec69cbc.pth",
    "vgg16_ssd_wider_face": "vgg16-ssd-wider_face-0_707-8c76d36acb083648.pth",
    "mb1_ssd_voc": "mb1-ssd-voc-mp-0_675-58694caf.pth",
    "mb2_ssd_lite_voc": "mb2-ssd-lite-voc-mp-0_686-b0d1ac2c.pth",
}


def ssd_model(model_name, dataset, create_model_fn, config, num_classes=20, pretrained=False,
    progress=True, device='cuda'):
    model = create_model_fn(num_classes + 1) # + 1 for background class
    model.config = config
    model.priors = config.priors.to(device)
    if pretrained:
        checkpoint_url = urlparse.urljoin(CHECKPOINT_STORAGE_URL, model_urls[f"{model_name}_{dataset}"])
        model = load_pretrained_weights(model, checkpoint_url, progress, device)

    return model.to(device)


def make_wrapper_func(wrapper_name, model_name, dataset, create_model_fn, config, num_classes):
    @MODEL_WRAPPER_REGISTRY.register(model_name=model_name, dataset_name=dataset,
        task_type='object_detection')
    def wrapper_func(pretrained=False, num_classes=num_classes, progress=True, device="cuda"):
        return ssd_model(
            model_name=model_name,
            dataset=dataset,
            create_model_fn=create_model_fn,
            config=config,
            num_classes=num_classes,
            pretrained=pretrained,
            progress=progress,
            device=device,
        )
    wrapper_func.__name__ = wrapper_name
    return wrapper_func


WrapperParams = namedtuple('WrapperParams', ['create_model_fn', 'config',
    'model_name', 'dataset', 'num_classes'])
MODEL_WRAPPERS = {
    'resnet18_ssd_voc': WrapperParams(create_model_fn=partial(create_resnet_ssd, backbone='resnet18'),
        config=VGG_CONFIG(), model_name='resnet18_ssd', dataset='voc', num_classes=20),
    'resnet34_ssd_voc': WrapperParams(create_model_fn=partial(create_resnet_ssd, backbone='resnet34'),
        config=VGG_CONFIG(), model_name='resnet34_ssd', dataset='voc', num_classes=20),
    'resnet50_ssd_voc': WrapperParams(create_model_fn=partial(create_resnet_ssd, backbone='resnet50'),
        config=VGG_CONFIG(), model_name='resnet50_ssd', dataset='voc', num_classes=20),
    'mb2_ssd_voc': WrapperParams(create_model_fn=create_mobilenetv2_ssd,
        config=MOBILENET_CONFIG(), model_name='mb2_ssd', dataset='voc', num_classes=20),
    'mb2_ssd_coco': WrapperParams(create_model_fn=create_mobilenetv2_ssd,
        config=MOBILENET_CONFIG(), model_name='mb2_ssd', dataset='coco', num_classes=80),
    'vgg16_ssd_voc': WrapperParams(create_model_fn=create_vgg_ssd,
        config=VGG_CONFIG(), model_name='vgg16_ssd', dataset='voc', num_classes=20),
    'vgg16_ssd_wider_face': WrapperParams(create_model_fn=create_vgg_ssd,
        config=VGG_CONFIG(), model_name='vgg16_ssd', dataset='wider_face', num_classes=1),
    'mb1_ssd_voc': WrapperParams(create_model_fn=create_mobilenetv1_ssd,
        config=MOBILENET_CONFIG(), model_name='mb1_ssd', dataset='voc', num_classes=20),
    'mb2_ssd_lite_voc': WrapperParams(create_model_fn=create_mobilenetv2_ssd_lite,
        config=MOBILENET_CONFIG(), model_name='mb2_ssd_lite', dataset='voc', num_classes=20),
}


for wrapper_fn_name, wrapper_params in MODEL_WRAPPERS.items():
    wrapper_fn_name = f'{wrapper_params.model_name}_{wrapper_params.dataset}'
    globals()[wrapper_fn_name] = make_wrapper_func(wrapper_fn_name, wrapper_params.model_name,
        wrapper_params.dataset, wrapper_params.create_model_fn, wrapper_params.config, wrapper_params.num_classes)
    __all__.append(wrapper_fn_name)
