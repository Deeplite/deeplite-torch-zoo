import torch
import torch.nn as nn


class MishCuda(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * torch.nn.functional.softplus(x).tanh()
