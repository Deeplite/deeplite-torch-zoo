import torch.nn as nn
from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct, DropPath


class LargeKernelReparam(nn.Module):
    def __init__(self, channels, kernel, small_kernel=5):
        super(LargeKernelReparam, self).__init__()
        self.dw_large = ConvBnAct(channels, channels, kernel, s=1, p=kernel//2,
            g=channels, act=None)
        self.small_kernel = small_kernel
        self.dw_small = ConvBnAct(channels, channels, small_kernel, s=1,
            p=small_kernel//2, g=channels, act=None)

    def forward(self, inp):
        outp = self.dw_large(inp)
        outp += self.dw_small(inp)
        return outp


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.,):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.fc1 = ConvBnAct(in_channels, hidden_features, 1,
            s=1, p=0, act=None)
        self.act = act_layer()
        self.fc2 = ConvBnAct(hidden_features, out_features, 1,
            s=1, p=0, act=None)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepLKBlock(nn.Module):

    def __init__(self, c1, c2, k=31, small_kernel=5, dw_ratio=1.0,
        mlp_ratio=4.0, drop_path=0., activation=nn.ReLU):
        super().__init__()

        self.pre_bn = nn.BatchNorm2d(c1)
        self.pw1 = ConvBnAct(c1, int(c1 * dw_ratio), 1, 1, 0, act=None)
        self.pw1_act = activation()
        self.dw = LargeKernelReparam(int(c1 * dw_ratio), k, small_kernel=small_kernel)
        self.dw_act = activation()
        self.pw2 = ConvBnAct(int(c1 * dw_ratio), c1, 1, 1, 0, act=None)

        self.premlp_bn = nn.BatchNorm2d(c1)
        self.mlp = MLP(in_channels=c1, hidden_channels=int(c1 * mlp_ratio))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y = self.pre_bn(x)
        y = self.pw1_act(self.pw1(y))
        y = self.dw_act(self.dw(y))
        y = self.pw2(y)
        x = x + self.drop_path(y)

        y = self.premlp_bn(x)
        y = self.mlp(y)
        x = x + self.drop_path(y)

        return x
