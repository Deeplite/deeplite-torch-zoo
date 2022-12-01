import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct, ACT_TYPE_MAP


class FireUnit(nn.Module) :
    """
    SqueezeNet unit, so-called 'Fire' unit.
    https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/squeezenet.py
    Parameters:
    ----------
    c1 : int
        Number of input channels.
    c2 : int
        Number of output channels.
    e : float , default 1/8
        Number of internal channels for squeeze convolution blocks.
    act : string
        Activation function to be used
    residual : bool
        Whether use residual connection.
    """
    def __init__(self,
                 c1,
                 c2,
                 e =1/8,
                 act = "relu",
                 residual = False):
        super(FireUnit, self).__init__()
        self.residual = residual
        in_channels = c1
        expand_channels = c2 // 2
        squeeze_channels = int(c2 * e) # Number of output channels for squeeze conv blocks.
        expand1x1_channels = expand_channels # Number of output channels for expand 1x1 conv blocks.
        expand3x3_channels = expand_channels # Number of output channels for expand 3x3 conv blocks.

        self.squeeze = ConvBnAct(
            c1=in_channels,
            c2=squeeze_channels,
            k=1,
            p=0,
            act=act,
            use_bn=False)
        self.expand1x1 = ConvBnAct(
            c1=squeeze_channels,
            c2=expand1x1_channels,
            k=1,
            p=0,
            act=act,
            use_bn=False)
        self.expand3x3 = ConvBnAct(
            c1=squeeze_channels,
            c2=expand3x3_channels,
            k=3,
            p=1,
            act=act,
            use_bn=False)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.squeeze(x)
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        out = torch.cat((y1, y2), dim=1)
        if self.residual:
            out = out + identity
        return out


class SqnxtUnit(nn.Module):
    """
    SqueezeNext unit.
    Parameters:
    ----------
    c1 : int
        Number of input channels.
    c2 : int
        Number of output channels.
    s : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 c1,
                 c2,
                 s = 1,
                 act="relu"):
        super(SqnxtUnit, self).__init__()
        stride = s

        self.resize_identity = True
        if stride == 2:
            reduction_den = 1
        elif c1 > c2:
            reduction_den = 4
        elif c1 < c2:
            reduction_den = 2
        else:
            reduction_den = 2
            self.resize_identity = False

        print ("self.resize_identity",self.resize_identity, reduction_den)
        # conv 1 x 1 block
        self.conv1 = ConvBnAct(
            c1=c1,
            c2=(c1 // reduction_den),
            k=1,
            s=stride,
            act=act)

        # conv 1 x 1 block
        self.conv2 = ConvBnAct(
            c1=(c1 // reduction_den),
            c2=(c1 // (2 * reduction_den)),
            k=1,
            act=act)

        # conv 1 x 3 block
        self.conv3 = ConvBnAct(
            c1=(c1 // (2 * reduction_den)),
            c2=(c1 // reduction_den),
            k=(1, 3),
            s=1,
            p=(0, 1),
            act=act)

        # conv 3 x 1 block
        self.conv4 = ConvBnAct(
            c1=(c1 // reduction_den),
            c2=(c1 // reduction_den),
            k=(3, 1),
            s=1,
            p=(1, 0),
            act=act)

        # conv 1 x 1 block
        self.conv5 = ConvBnAct(
            c1=(c1 // reduction_den),
            c2=c2,
            k=1,
            act=act)

        if self.resize_identity:
            self.identity_conv = ConvBnAct(
                c1=c1,
                c2=c2,
                s=stride,
                act=act)
        self.activ = ACT_TYPE_MAP[act] if act else nn.Identity()

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        print(identity.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + identity
        x = self.activ(x)
        return x



def _test_blocks(c1, c2, b=2, res=32):
    # perform a basic forward pass
    input = torch.rand((b,c1,res, res), device=None, requires_grad=False)

    fire_block = FireUnit(c1, c2)
    output = fire_block(input)
    assert output.shape == (b, c2, res, res)

    sqnxrt_block = SqnxtUnit(c1, c2)
    output = sqnxrt_block(input)
    assert output.shape == (b, c2, res, res)

if __name__ == "__main__":

    _test_blocks(64, 64) #  c1 == c2
    _test_blocks(64, 32) #  c1 > c2
    _test_blocks(32, 64) #  c1 < c2
