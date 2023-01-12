import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    


class GhostModuleV2(nn.Module):
    def __init__(self, c1, c2, k=3, ratio=2, dw_size=3, s=1, relu=True,mode=None):
        super(GhostModuleV2, self).__init__()
        self.mode=mode
        self.gate_fn=nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = c2
            init_channels = math.ceil(c2 / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(c1, init_channels, k, s, k//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']: 
            self.oup = c2
            init_channels = math.ceil(c2 / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(c1, init_channels, k, s, k//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            ) 
            self.short_conv = nn.Sequential( 
                nn.Conv2d(c1, c2, k, s, k//2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=(1,5), stride=1, padding=(0,2), groups=c2,bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=(5,1), stride=1, padding=(2,0), groups=c2,bias=False),
                nn.BatchNorm2d(c2),
            ) 

    def attention_layer(self, x, downscale=True):

        res = F.avg_pool2d(x,kernel_size=2,stride=2) #if downscale else x
        res=self.short_conv(res)
        res = self.gate_fn(res)
        if downscale:
            res = F.interpolate(res,size=x.shape[-1],mode='nearest')
        return res

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        if self.mode in ['original']:
            return out[:,:self.oup,:,:]
        elif self.mode in ['attn']:
            return out[:,:self.oup,:,:]*self.attention_layer(x)


def _test_blocks(c1, c2, b=2, res=32):
    input = torch.rand((b,c1,res, res), device=None, requires_grad=False)

    block = GhostModuleV2(c1, c2, mode="original")
    output = block(input)
    assert output.shape == (b, c2, res, res)

    block = GhostModuleV2(c1, c2, mode="attn")
    output = block(input)
    assert output.shape == (b, c2, res, res)


if __name__ == "__main__":
    _test_blocks(64, 64) #  c1 == c2
    _test_blocks(64, 32) #  c1 > c2
    _test_blocks(32, 64) #  c1 < c2
