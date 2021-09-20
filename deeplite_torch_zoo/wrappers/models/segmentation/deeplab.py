from torch.hub import load_state_dict_from_url

from deeplite_torch_zoo.src.segmentation.deeplab.repo.modeling.deeplab import DeepLab


__all__ = ["deeplab_mobilenet_voc_20"]


model_urls = {
    "mobilenet_voc_20": "http://download.deeplite.ai/zoo/models/deeplab-mobilenet-voc-20_593-94ac51da679409d6.pth",
}


def deeplab(
    backbone="resnet",
    dataset="voc_20",
    num_classes=21,
    pretrained=False,
    progress=True,
    device="cuda",
):
    model = DeepLab(backbone=backbone, num_classes=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[f"{backbone}_{dataset}"],
            progress=progress,
            check_hash=True,
            map_location=device
        )
        model.load_state_dict(state_dict)
    return model.to(device)


def deeplab_mobilenet_voc_20(pretrained=True, progress=False, device='cuda'):
    return deeplab(
        backbone="mobilenet",
        dataset="voc_20",
        num_classes=21,
        pretrained=pretrained,
        progress=progress,
        device=device,
    )
