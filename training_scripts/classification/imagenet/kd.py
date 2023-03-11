# Source: https://github.com/Alibaba-MIIL/Solving_ImageNet/blob/main/kd/kd_utils.py

import torch
import torch.nn as nn
import torchvision.transforms as T

from deeplite_torch_zoo import create_model


class KDTeacher(nn.Module):
    def __init__(self, args=None, handle_inplace_abn=False):
        super(KDTeacher, self).__init__()

        model_kd = create_model(
            model_name=args.kd_model_name,
            pretraining_dataset='imagenet',
            pretrained=True,
            num_classes=args.num_classes,
        )

        if args.kd_model_checkpoint is not None:
            model_kd.load_state_dict(torch.load(args.kd_model_checkpoint))

        model_kd.cpu().eval()

        if handle_inplace_abn:
            model_kd = InplacABN_to_ABN(model_kd)
            model_kd = fuse_bn2d_bn1d_abn(model_kd)

        self.model = model_kd.cuda().eval()

        self.mean_model_kd = None
        self.std_model_kd = None
        if hasattr(model_kd, 'default_cfg'):
            self.mean_model_kd = model_kd.default_cfg['mean']
            self.std_model_kd = model_kd.default_cfg['std']

    # handling different normalization of teacher and student
    def normalize_input(self, input, student_model):
        if hasattr(student_model, 'module'):
            model_s = student_model.module
        else:
            model_s = student_model

        input_kd = input

        if hasattr(model_s, 'default_cfg'):
            mean_student = model_s.default_cfg['mean']
            std_student = model_s.default_cfg['std']

            if mean_student != self.mean_model_kd or std_student != self.std_model_kd:
                std = (self.std_model_kd[0] / std_student[0], self.std_model_kd[1] / std_student[1],
                    self.std_model_kd[2] / std_student[2])
                transform_std = T.Normalize(mean=(0, 0, 0), std=std)

                mean = (self.mean_model_kd[0] - mean_student[0], self.mean_model_kd[1] - mean_student[1],
                        self.mean_model_kd[2] - mean_student[2])
                transform_mean = T.Normalize(mean=mean, std=(1, 1, 1))

                input_kd = transform_mean(transform_std(input))

        return input_kd


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)



def fuse_bn_to_conv(bn_layer, conv_layer):
    bn_st_dict = bn_layer.state_dict()
    conv_st_dict = conv_layer.state_dict()

    # BatchNorm params
    eps = bn_layer.eps
    mu = bn_st_dict['running_mean']
    var = bn_st_dict['running_var']
    gamma = bn_st_dict['weight']

    if 'bias' in bn_st_dict:
        beta = bn_st_dict['bias']
    else:
        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

    # Conv params
    W = conv_st_dict['weight']
    if 'bias' in conv_st_dict:
        bias = conv_st_dict['bias']
    else:
        bias = torch.zeros(W.size(0)).float().to(gamma.device)

    denom = torch.sqrt(var + eps)
    b = beta - gamma.mul(mu).div(denom)
    A = gamma.div(denom)
    bias *= A
    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

    W.mul_(A)
    bias.add_(b)

    conv_layer.weight.data.copy_(W)
    if conv_layer.bias is None:
        conv_layer.bias = torch.nn.Parameter(bias)
    else:
        conv_layer.bias.data.copy_(bias)


def fuse_bn_to_linear(bn_layer, linear_layer):
    # print('bn fuse')
    bn_st_dict = bn_layer.state_dict()
    conv_st_dict = linear_layer.state_dict()

    # BatchNorm params
    eps = bn_layer.eps
    mu = bn_st_dict['running_mean']
    var = bn_st_dict['running_var']
    gamma = bn_st_dict['weight']

    if 'bias' in bn_st_dict:
        beta = bn_st_dict['bias']
    else:
        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

    # Conv params
    W = conv_st_dict['weight']
    if 'bias' in conv_st_dict:
        bias = conv_st_dict['bias']
    else:
        bias = torch.zeros(W.size(0)).float().to(gamma.device)

    denom = torch.sqrt(var + eps)
    b = beta - gamma.mul(mu).div(denom)
    A = gamma.div(denom)
    bias *= A
    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

    W.mul_(A)
    bias.add_(b)

    linear_layer.weight.data.copy_(W)
    if linear_layer.bias is None:
        linear_layer.bias = torch.nn.Parameter(bias)
    else:
        linear_layer.bias.data.copy_(bias)


def extract_layers(model):
    list_layers = []
    for n, p in model.named_modules():
        list_layers.append(n)
    return list_layers


def compute_next_bn(layer_name, resnet):
    list_layer = extract_layers(resnet)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_bn = list_layer[list_layer.index(layer_name) + 1]
    if extract_layer(resnet, next_bn).__class__.__name__ == 'BatchNorm2d':
        return next_bn
    return None


def compute_next_abn(layer_name, resnet):
    list_layer = extract_layers(resnet)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_bn = list_layer[list_layer.index(layer_name) + 1]
    if extract_layer(resnet, next_bn).__class__.__name__ == 'ABN':
        return next_bn
    return None


def compute_next_bn_1d(layer_name, resnet):
    list_layer = extract_layers(resnet)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_bn = list_layer[list_layer.index(layer_name) + 1]
    if extract_layer(resnet, next_bn).__class__.__name__ == 'BatchNorm1d':
        return next_bn
    return None


def fuse_bn2d_bn1d_abn(model):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            next_bn = compute_next_bn(n, model)
            if next_bn is not None:
                next_bn_ = extract_layer(model, next_bn)
                fuse_bn_to_conv(next_bn_, m)
                set_layer(model, next_bn, nn.Identity())

            next_abn = compute_next_abn(n, model)
            if next_abn is not None:
                next_bn_ = extract_layer(model, next_abn)
                activation = calc_abn_activation(next_bn_)
                fuse_bn_to_conv(next_bn_, m)
                set_layer(model, next_abn, activation)
        if isinstance(m, torch.nn.Linear):
            next_bn1d = compute_next_bn_1d(n, model)
            if next_bn1d is not None:
                next_bn1d_ = extract_layer(model, next_bn1d)
                fuse_bn_to_linear(next_bn1d_, m)
                set_layer(model, next_bn1d, nn.Identity())

    return model


def calc_abn_activation(ABN_layer):
    from inplace_abn import ABN
    activation = nn.Identity()
    if isinstance(ABN_layer, ABN):
        if ABN_layer.activation == "relu":
            activation = nn.ReLU(inplace=True)
        elif ABN_layer.activation == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=ABN_layer.activation_param, inplace=True)
        elif ABN_layer.activation == "elu":
            activation = nn.ELU(alpha=ABN_layer.activation_param, inplace=True)
    return activation


def InplacABN_to_ABN(module: nn.Module) -> nn.Module:
    from inplace_abn import ABN
    from timm.models.layers import InplaceAbn

    # convert all InplaceABN layer to bit-accurate ABN layers.
    if isinstance(module, InplaceAbn):
        module_new = ABN(module.num_features, activation=module.act_name,
                         activation_param=module.act_param)
        for key in module.state_dict():
            module_new.state_dict()[key].copy_(module.state_dict()[key])
        module_new.training = module.training
        module_new.weight.data = module_new.weight.abs() + module_new.eps
        return module_new
    for name, child in reversed(module._modules.items()):
        new_child = InplacABN_to_ABN(child)
        if new_child != child:
            module._modules[name] = new_child
    return module
