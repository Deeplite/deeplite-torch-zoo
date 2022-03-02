
from torchvision import models
from deeplite_torch_zoo.wrappers.models.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["resnet18_tinyimagenet", "resnet34_tinyimagenet", "resnet50_tinyimagenet"]


model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18_tinyimagenet_0_663-b0637203dfcca31b.pt",
    "resnet34": "http://download.deeplite.ai/zoo/models/resnet34_tinyimagenet_0_6863-698d71e10fe153f0.pt",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50_tinyimagenet_0_7303-8ec06f70f32c110b.pt",
}


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet18', dataset_name='tinyimagenet', task_type='classification')
def resnet18_tinyimagenet(pretrained=False, progress=True, num_classes=100, device="cuda"):
    model = models.resnet18(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls['resnet18']
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet34', dataset_name='tinyimagenet', task_type='classification')
def resnet34_tinyimagenet(pretrained=False, progress=True, num_classes=100, device="cuda"):
    model = models.resnet34(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls['resnet34']
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


@MODEL_WRAPPER_REGISTRY.register(model_name='resnet50', dataset_name='tinyimagenet', task_type='classification')
def resnet50_tinyimagenet(pretrained=False, progress=True, num_classes=100, device="cuda"):
    model = models.resnet50(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls['resnet50']
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)
