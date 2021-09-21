from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.segmentation.unet_scse.repo.src.models.net import EncoderDecoderNet
from deeplite_torch_zoo.src.segmentation.Unet.model.unet_model import UNet

__all__ = [
    "unet_enc_dec",
    "unet_carvana",
    "unet_scse_resnet18_voc_20",
    "unet_scse_resnet18_voc_1",
    "unet_scse_resnet18_voc_2",
    "unet_scse_resnet18_carvana",
]


model_urls = {
    "unet_carvana": "http://download.deeplite.ai/zoo/models/unet-carvana-dc_994-bb5835e8c29769c0.pth",
    "unet_scse_resnet18_carvana": "http://download.deeplite.ai/zoo/models/unet_scse_resnet18-carvana-1cls-0_989-077cf500850096ba.pth",
    "unet_scse_resnet18_voc_20": "http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-miou_593-1e0987c833e9abd7.pth",
    "unet_scse_resnet18_voc_01": "http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-1cls-0_682-38cbf3aaa2ce9a46.pth",
    "unet_scse_resnet18_voc_02": "http://download.deeplite.ai/zoo/models/unet_scse_resnet18-voc-2cls-0_688-79087739621c42c1.pth",
}


def unet_carvana(pretrained=False, progress=True, device="cuda"):
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["unet_carvana"], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)
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
        state_dict = load_state_dict_from_url(
            model_urls[f"{dec_type}_{enc_type}_{dataset_type}"],
            progress=progress,
            check_hash=True,
            map_location=device,
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def unet_scse_resnet18_voc_20(pretrained=True, progress=False, device="cuda"):
    return unet_enc_dec(
        enc_type="resnet18",
        dec_type="unet_scse",
        output_channels=21,
        dataset_type="voc_20",
        num_filters=8,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def unet_scse_resnet18_voc_1(pretrained=True, progress=False, device="cuda"):
    return unet_enc_dec(
        enc_type="resnet18",
        dec_type="unet_scse",
        output_channels=1,
        dataset_type="voc_01",
        num_filters=8,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def unet_scse_resnet18_voc_2(pretrained=True, progress=False, device="cuda"):
    return unet_enc_dec(
        enc_type="resnet18",
        dec_type="unet_scse",
        output_channels=3,
        dataset_type="voc_02",
        num_filters=8,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )


def unet_scse_resnet18_carvana(pretrained=True, progress=False, device="cuda"):
    return unet_enc_dec(
        enc_type="resnet18",
        dec_type="unet_scse",
        output_channels=1,
        dataset_type="carvana",
        num_filters=8,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
