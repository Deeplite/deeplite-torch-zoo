from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.src.classification.cifar_models.mobileone import mobileone


__all__ = ["mobileone_s0", "mobileone_s1", "mobileone_s2", "mobileone_s3", "mobileone_s4"]

model_urls = {
    "s0": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s0.pth.tar",
    "s1": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s1.pth.tar",
    "s2": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s2.pth.tar",
    "s3": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s3.pth.tar",
    "s4": "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s4.pth.tar",
}

def get_mobileone_variant(pretrained=False, progress=True, device="cuda", num_classes=1000, variant="s0"):
    model = mobileone(num_classes=num_classes, inference_mode = not pretrained, variant=variant )
    if pretrained:
        checkpoint_url = model_urls[variant]
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s0', dataset_name='imagenet', task_type='classification')
def mobileone_s0(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s0")

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s1', dataset_name='imagenet', task_type='classification')
def mobileone_s1(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s1")

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s2', dataset_name='imagenet', task_type='classification')
def mobileone_s2(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s2")

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s3', dataset_name='imagenet', task_type='classification')
def mobileone_s3(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s3")

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s4', dataset_name='imagenet', task_type='classification')
def mobileone_s4(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s4")
