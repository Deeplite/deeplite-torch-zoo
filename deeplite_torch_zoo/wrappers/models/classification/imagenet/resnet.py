import torchvision.models as models


__all__ = ["resnet18_imagenet", "resnet34_imagenet", "resnet50_imagenet"]


def resnet18_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet18(pretrained=pretrained)
    return model.to(device)


def resnet34_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet34(pretrained=pretrained)
    return model.to(device)


def resnet50_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet50(pretrained=pretrained)
    return model.to(device)
