from torch.hub import load_state_dict_from_url

__all__ = ["mb1_ssd_voc_20"]

from deeplite_torch_zoo.src.objectdetection.ssd.repo.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from deeplite_torch_zoo.src.objectdetection.ssd.config.mobilenetv1_ssd_config import MOBILENET_CONFIG


model_urls = {
    "mb1_ssd": "http://download.deeplite.ai/zoo/models/mb1-ssd-voc-mp-0_675-58694caf.pth",
}


def mb1_ssd_voc_20(num_classes=20, pretrained=False, progress=True, device='cuda'):
    model = create_mobilenetv1_ssd(num_classes + 1)
    config = MOBILENET_CONFIG()
    model.config = config
    model.priors = config.priors.to(device)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["mb1_ssd"], progress=progress, check_hash=True, map_location=device
        )
        model.load_state_dict(state_dict)

    return model
