from deeplite_torch_zoo.src.segmentation.unet_scse.repo.src.models.net import EncoderDecoderNet
from deeplite_torch_zoo.src.segmentation.Unet.model.unet_model import UNet
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights


__all__ = [
    "unet_enc_dec",
    "unet_carvana",
    "unet_scse_resnet18_voc",
    "unet_scse_resnet18_carvana",
]


model_urls = {
    "unet_carvana": "http://download.deeplite.ai/zoo/models/unet-carvana-dc_994-bb5835e8c29769c0.pth",
    "unet_scse_resnet18_carvana": "http://download.deeplite.ai/zoo/models/unet_scse_resnet18-carvana-1cls-0_989-077cf500850096ba.pth",
    "unet_scse_resnet18_voc": "http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-miou_593-1e0987c833e9abd7.pth",
}


@MODEL_WRAPPER_REGISTRY.register(model_name='unet', dataset_name='carvana',
    task_type='semantic_segmentation')
def unet_carvana(pretrained=False, progress=True, num_classes=1, device="cuda"):
    model = UNet(n_channels=3, n_classes=num_classes, bilinear=True)
    if pretrained:
        checkpoint_url = model_urls["unet_carvana"]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def unet_enc_dec(
    enc_type="resnet50",
    dec_type="unet_scse",
    output_channels=21,
    dataset_type="voc",
    num_filters=16,
    pretrained=True,
    progress=False,
    device="cuda",
):
    model = EncoderDecoderNet(
        output_channels=output_channels,
        enc_type=enc_type,
        dec_type=dec_type,
        num_filters=num_filters,
    )
    if pretrained:
        checkpoint_url = model_urls[f"{dec_type}_{enc_type}_{dataset_type}"]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)

    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='unet_scse_resnet18', dataset_name='voc',
    task_type='semantic_segmentation')
def unet_scse_resnet18_voc(pretrained=True, progress=False, num_classes=20, device="cuda"):
    return unet_enc_dec(
        enc_type="resnet18",
        dec_type="unet_scse",
        output_channels=num_classes+1,
        dataset_type="voc",
        num_filters=8,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


@MODEL_WRAPPER_REGISTRY.register(model_name='unet_scse_resnet18', dataset_name='carvana',
    task_type='semantic_segmentation')
def unet_scse_resnet18_carvana(pretrained=True, progress=False, num_classes=1, device="cuda"):
    return unet_enc_dec(
        enc_type="resnet18",
        dec_type="unet_scse",
        output_channels=num_classes,
        dataset_type="carvana",
        num_filters=8,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
