# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

from copy import deepcopy
from pathlib import Path

import inspect

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks import *

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv
from deeplite_torch_zoo.src.object_detection.yolo.common import *
from deeplite_torch_zoo.src.object_detection.yolo.experimental import *
from deeplite_torch_zoo.src.object_detection.yolo.heads import (
    Detect, DetectV8, DetectX
)
from deeplite_torch_zoo.src.object_detection.yolo.yolov5 import DetectionModel

from deeplite_torch_zoo.utils import (
    LOGGER,
    make_divisible,
    initialize_weights,
)
from deeplite_torch_zoo.src.registries import EXPANDABLE_BLOCKS, VARIABLE_CHANNEL_BLOCKS


HEAD_NAME_MAP = {
    'v5': Detect,
    'v8': DetectV8,
    'x': DetectX,
}


class YOLO(DetectionModel):
    # Modified YOLOv5 version 6 taken from
    # commit 15e8c4c15bff0 at https://github.com/ultralytics/yolov5
    def __init__(
        self,
        cfg='yolov5s.yaml',
        ch=3,
        nc=None,
        anchors=None,
        activation_type=None,
        depth_mul=None,
        width_mul=None,
        channel_divisor=8,
        max_channels=None,
        custom_head=None,
        verbose=False,
    ):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        if custom_head is not None:
            self.yaml['head'][-1][2] = HEAD_NAME_MAP[custom_head].__name__

        # Define model
        self.nc = nc
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml),
            ch=[ch],
            activation_type=activation_type,
            depth_mul=depth_mul,
            width_mul=width_mul,
            yolo_channel_divisor=channel_divisor,
            max_channels=max_channels,
        )  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        self._init_head(ch)

        # Init weights, biases
        initialize_weights(self)
        self._is_fused = False
        if verbose:
            self.info()


def parse_model(
    d,
    ch,
    activation_type,
    depth_mul=None,
    width_mul=None,
    max_channels=None,
    yolo_channel_divisor=8,
):
    LOGGER.info(
        f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"
    )

    anchors, nc = d['anchors'], d['nc']
    gd = depth_mul
    gw = width_mul
    activation_type = (
        activation_type if activation_type is not None else d['activation_type']
    )

    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d['backbone'] + d['head']
    ):  # from, number, module, args
        try:
            m = eval(m) if isinstance(m, str) else m  # eval strings
        except:
            m = eval(f'YOLO{m}') if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in VARIABLE_CHANNEL_BLOCKS.registry_dict.values():
            c1, c2 = ch[f], args[0]

            if c2 != no:  # if not output
                c2 = make_divisible(min(c2, max_channels) * gw, yolo_channel_divisor)

            args = [c1, c2, *args[1:]]
            if m in EXPANDABLE_BLOCKS.registry_dict.values():
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is ADD or m is Shortcut:
            c2 = ch[f[0]]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is DetectX:
            args.append([ch[x] for x in f])
        elif m is DetectV8:
            args = args[:1]
            args.append([ch[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is ReOrg or m is DWT:
            c2 = ch[f] * 4
        else:
            c2 = ch[f]

        kwargs = dict()
        if 'act' in inspect.signature(m).parameters:
            kwargs.update({'act': activation_type})

        m_ = (
            nn.Sequential(*(m(*args, **kwargs) for _ in range(n)))
            if n > 1
            else m(*args, **kwargs)
        )  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        LOGGER.info(
            f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}'
        )  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
