"""

[1] Implementation
    https://github.com/aaron-xichen/pytorch-playground

"""

from collections import OrderedDict

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers["fc{}".format(i + 1)] = nn.Linear(current_dims, n_hidden)
            layers["relu{}".format(i + 1)] = nn.ReLU()
            layers["drop{}".format(i + 1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers["out"] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.model.forward(input)
