# Code credit: https://github.com/Bobo-y/flexible-yolov5

import torch
import yaml
from addict import Dict
from torch import nn

from deeplite_torch_zoo.src.object_detection.flexible_yolo.backbone import (
    build_backbone,
)
from deeplite_torch_zoo.src.object_detection.flexible_yolo.neck import build_neck
from deeplite_torch_zoo.src.object_detection.yolov5.heads.detect import Detect
from deeplite_torch_zoo.src.object_detection.yolov5.heads.detect_v8 import (
    DetectV8,
)
from deeplite_torch_zoo.src.object_detection.yolov5.yolov5 import (
    HEAD_NAME_MAP,
    Conv,
    DWConv,
    RepConv,
    YOLOModel,
    fuse_conv_and_bn,
)
from deeplite_torch_zoo.utils import initialize_weights, LOGGER


class FlexibleYOLO(YOLOModel):
    def __init__(
        self,
        model_config,
        nc=None,
        backbone_kwargs=None,
        neck_kwargs=None,
        custom_head=None,
    ):
        """
        :param model_config:
        """
        nn.Module.__init__(self)
        self.yaml = None

        head_cls = Detect
        if custom_head is not None:
            head_cls = HEAD_NAME_MAP[custom_head]

        if type(model_config) is str:
            model_config = yaml.load(open(model_config, 'r'), Loader=yaml.SafeLoader)
        model_config = Dict(model_config)
        if nc is not None:
            model_config.head.nc = nc

        if backbone_kwargs is not None:
            model_config.backbone.update(Dict(backbone_kwargs))

        backbone_type = model_config.backbone.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        ch_in = self.backbone.out_shape

        self.necks = nn.ModuleList()
        necks_config = model_config.neck
        if neck_kwargs is not None:
            necks_config.update(Dict(neck_kwargs))

        for neck_name, neck_params in necks_config.items():
            neck_params['ch'] = ch_in
            neck = build_neck(neck_name, **neck_params)
            ch_in = neck.out_shape
            self.necks.append(neck)

        model_config.head['ch'] = ch_in

        if head_cls != Detect:
            model_config.head.pop('anchors')
        self.detection = head_cls(**model_config.head)

        self._init_head()

        initialize_weights(self)
        self._is_fused = False

    def _init_head(self):
        if isinstance(self.detection, Detect):
            s = 256  # 2x min stride
            self.detection.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, 3, s, s))]
            )
            self.detection.anchors /= self.detection.stride.view(-1, 1, 1)

            self.stride = self.detection.stride
            self._initialize_biases()  # only run once
        if isinstance(self.detection, DetectV8):
            s = 256  # 2x min stride
            # self.detection.inplace = self.inplace
            forward = lambda x: self.forward(x)
            self.detection.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))]
            )  # forward
            self.stride = self.detection.stride
            self.detection.bias_init()  # only run once

    def forward(self, x):
        out = self.backbone(x)
        for neck in self.necks:
            out = neck(out)
        y = self.detection(list(out))
        return y

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = nn.Module._apply(self, fn)
        m = self.detection  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    def fuse(self, verbose=False):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
        self.info()
        self._is_fused = True
        return self

    def is_fused(self):
        return self._is_fused
