#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# Taken from https://github.com/apple/ml-mobileone

import copy
from typing import Tuple

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.cnn_attention import SELayer
from deeplite_torch_zoo.src.dnn_blocks.common import get_activation


class MobileOneBlock(nn.Module):
    """ MobileOne building block.
        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 c1: int,
                 c2: int,
                 k: int = 3,
                 s: int = 1,
                 p: int = 1,
                 d: int = 1,
                 g: int = 1,
                 act="relu",
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.

        :param c1: Number of channels in the input.
        :param c2: Number of channels produced by the block.
        :param k: Size of the convolution kernel.
        :param s: Stride size.
        :param p: Zero-padding size.
        :param d: Kernel dilation factor.
        :param g: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = g
        self.stride = s
        self.kernel_size = k
        self.in_channels = c1
        self.out_channels = c2
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        self.se = SELayer(c2) if use_se else nn.Identity()
        self.activation = get_activation(act)

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=c1,
                                          out_channels=c2,
                                          kernel_size=k,
                                          stride=s,
                                          padding=p,
                                          dilation=d,
                                          groups=g,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=c1) \
                if c2 == c1 and s == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=k,
                                              padding=p))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if k > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileOneBlockUnit(nn.Module):
    """ MobileOne building block,
        It is a combination of Depthwise and pointwise conv, as mentioned in paper

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 c1: int,
                 c2: int,
                 k: int = 3,
                 s: int = 1,
                 p: int = 1,
                 d: int = 1,
                 g: int = 1,
                 act="relu",
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.
        Mobile block builds on the MobileNet-V1 [1] block of 3x3 depthwise convolution followed
        by 1x1 pointwise convolutions
        :param c1: Number of channels in the input.
        :param c2: Number of channels produced by the block.
        :param k: Size of the convolution kernel.
        :param s: Stride size.
        :param p: Zero-padding size.
        :param d: Kernel dilation factor.
        :param g: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()
        self.depthwise_conv = MobileOneBlock(c1=c1,
                                             c2=c1,
                                             k=k,
                                             s=s,
                                             p=p,
                                             d=d,
                                             g=c1,
                                             act=act,
                                             inference_mode=inference_mode,
                                             use_se=use_se,
                                             num_conv_branches=num_conv_branches )

        self.pointwise_conv = MobileOneBlock(c1=c1,
                                             c2=c2,
                                             k=1,
                                             s=s,
                                             p=0,
                                             d=d,
                                             g=g,
                                             act=act,
                                             inference_mode=inference_mode,
                                             use_se=use_se,
                                             num_conv_branches=num_conv_branches )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model


def _calc_width(net):
    import numpy as np
    net_params = net.parameters()#filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test_reparametrize_model(c1, c2, b=2, res=32):
    # perform a basic forward pass after reparam
    input = torch.rand((b,c1,res, res), device=None, requires_grad=False)
    block = MobileOneBlock(c1, c2, 3)
    weight_count1 = _calc_width(block)
    block = reparameterize_model(block)
    weight_count2 = _calc_width(block)
    output = block(input) # test forward pass after reparam
    print (f"Weight Before and after reparameterization {weight_count1} -> {weight_count2}")

    block = MobileOneBlockUnit(c1, c2, 3)
    weight_count1 = _calc_width(block)
    block = reparameterize_model(block)
    weight_count2 = _calc_width(block)
    output = block(input) # test forward pass after reparam
    print (f"Weight Before and after reparameterization {weight_count1} -> {weight_count2}")


def _test_blocks(c1, c2, b=2, res=32):
    # perform a basic forward pass
    input = torch.rand((b,c1,res, res), device=None, requires_grad=False)

    block = MobileOneBlock(c1, c2, 3)
    output = block(input)
    assert output.shape == (b, c2, res, res)

    block = MobileOneBlockUnit(c1, c2, 3)
    output = block(input)
    assert output.shape == (b, c2, res, res)



if __name__ == "__main__":
    _test_blocks(64, 64) #  c1 == c2
    _test_blocks(64, 32) #  c1 > c2
    _test_blocks(32, 64) #  c1 < c2

    _test_reparametrize_model(64, 64, res=16)
