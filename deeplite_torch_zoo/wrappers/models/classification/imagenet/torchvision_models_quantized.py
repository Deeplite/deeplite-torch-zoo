import torchvision.models.quantization as models


def q_resnet18_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet18(pretrained=pretrained)
    return model.to(device)


def q_resnet50_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet50(pretrained=pretrained)
    return model.to(device)


def q_googlenet_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.googlenet(pretrained=pretrained)
    return model.to(device)


def q_shufflenet_v2_x0_5_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.shufflenet_v2_x0_5(pretrained=pretrained)
    return model.to(device)


def q_shufflenet_v2_x1_0_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    return model.to(device)


def q_mobilenet_v2_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.mobilenet_v2(pretrained=pretrained)
    return model.to(device)


def q_resnext101_32x8d_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnext101_32x8d(pretrained=pretrained)
    return model.to(device)


def q_inception_v3_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.inception_v3(pretrained=pretrained)
    return model.to(device)
