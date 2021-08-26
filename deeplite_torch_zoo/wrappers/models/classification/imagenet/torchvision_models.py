from torchvision import models

def resnet18_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet18(pretrained=pretrained)
    return model.to(device)


def resnet34_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet34(pretrained=pretrained)
    return model.to(device)


def resnet50_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet50(pretrained=pretrained)
    return model.to(device)


def resnet101_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet101(pretrained=pretrained)
    return model.to(device)


def resnet152_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnet152(pretrained=pretrained)
    return model.to(device)


def alexnet_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.alexnet(pretrained=pretrained)
    return model.to(device)


def vgg16_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.vgg16(pretrained=pretrained)
    return model.to(device)


def squeezenet1_0_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.squeezenet1_0(pretrained=pretrained)
    return model.to(device)


def squeezenet1_1_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.squeezenet1_1(pretrained=pretrained)
    return model.to(device)


def densenet121_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.densenet121(pretrained=pretrained)
    return model.to(device)


def densenet161_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.densenet161(pretrained=pretrained)
    return model.to(device)


def densenet169_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.densenet169(pretrained=pretrained)
    return model.to(device)


def densenet201_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.densenet201(pretrained=pretrained)
    return model.to(device)


def googlenet_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.googlenet(pretrained=pretrained)
    return model.to(device)


def shufflenet_v2_x0_5_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.shufflenet_v2_x0_5(pretrained=pretrained)
    return model.to(device)


def shufflenet_v2_x1_0_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    return model.to(device)


def mobilenet_v2_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.mobilenet_v2(pretrained=pretrained)
    return model.to(device)


def resnext50_32x4d_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnext50_32x4d(pretrained=pretrained)
    return model.to(device)


def resnext101_32x8d_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.resnext101_32x8d(pretrained=pretrained)
    return model.to(device)


def wide_resnet50_2_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.wide_resnet50_2(pretrained=pretrained)
    return model.to(device)


def wide_resnet101_2_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.wide_resnet101_2(pretrained=pretrained)
    return model.to(device)


def mnasnet1_0_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.mnasnet1_0(pretrained=pretrained)
    return model.to(device)


def mnasnet0_5_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.mnasnet0_5(pretrained=pretrained)
    return model.to(device)


def vgg11_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.vgg11(pretrained=pretrained)
    return model.to(device)


def vgg11_bn_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.vgg11_bn(pretrained=pretrained)
    return model.to(device)


def vgg13_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.vgg13(pretrained=pretrained)
    return model.to(device)


def vgg13_bn_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.vgg13_bn(pretrained=pretrained)
    return model.to(device)


def vgg16_bn_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.vgg16_bn(pretrained=pretrained)
    return model.to(device)


def vgg19_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.vgg19(pretrained=pretrained)
    return model.to(device)


def vgg19_bn_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.vgg19_bn(pretrained=pretrained)
    return model.to(device)


def inception_v3_imagenet(pretrained=False, progress=True, device="cuda"):
    model = models.inception_v3(pretrained=pretrained)
    return model.to(device)
