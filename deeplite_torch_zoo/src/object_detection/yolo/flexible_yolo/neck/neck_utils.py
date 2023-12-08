
NECK_ACT_TYPE_MAP = {
    'relu': dict(type='ReLU', inplace=True),
    'relu6': dict(type='ReLU6', inplace=True),
    'hswish': dict(type='Hardswish', inplace=True),
    'hardswish': dict(type='Hardswish', inplace=True),
    'silu': dict(type='SiLU', inplace=True),
    'hsigmoid': dict(type='Hardsigmoid', inplace=True),
    'sigmoid': dict(type='Sigmoid'),
    'lrelu': dict(type='LeakyReLU', negative_slope=0.1, inplace=True),
    'leakyrelu': dict(type='LeakyReLU', negative_slope=0.1, inplace=True),
    'leakyrelu_0.1': dict(type='LeakyReLU', negative_slope=0.1, inplace=True),
    'gelu': dict(type='GELU'),
}


YOLO_SCALING_GAINS = {
    'n': {'gd': 0.33, 'gw': 0.25},
    's': {'gd': 0.33, 'gw': 0.5},
    'm': {'gd': 0.67, 'gw': 0.75},
    'l': {'gd': 1, 'gw': 1},
    'x': {'gd': 1.33, 'gw': 1.25},
}
