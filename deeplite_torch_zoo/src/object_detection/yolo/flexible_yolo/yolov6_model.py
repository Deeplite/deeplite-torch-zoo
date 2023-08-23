import torch.nn as nn

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.model import FlexibleYOLO
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6 import build_network
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6.config import Config
from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6.layers.common import (
    RepVGGBlock,
)

from deeplite_torch_zoo.src.object_detection.yolo.heads import Detect
from deeplite_torch_zoo.src.object_detection.yolo.anchors import ANCHOR_REGISTRY
from deeplite_torch_zoo.src.object_detection.yolo.yolov5 import (
    Conv,
    DWConv,
    RepConv,
    fuse_conv_and_bn,
)
from deeplite_torch_zoo.src.object_detection.yolo.config_parser import HEAD_NAME_MAP

from deeplite_torch_zoo.utils import initialize_weights, LOGGER


class YOLOv6(FlexibleYOLO):
    def __init__(
        self, model_config, nc=80, anchors=None, custom_head=None, width_mul=None, depth_mul=None
    ):
        """
        :param model_config:
        """
        nn.Module.__init__(self)

        head_config = {
            'nc': nc,
            'anchors': anchors if anchors is not None \
                else ANCHOR_REGISTRY.get('default')(),
        }

        head_cls = Detect
        if custom_head is not None:
            head_cls = HEAD_NAME_MAP[custom_head]

        cfg = Config.fromfile(model_config)
        if not hasattr(cfg, 'training_mode'):
            setattr(cfg, 'training_mode', 'repvgg')

        if width_mul is not None:
            cfg.model.width_multiple = width_mul
        if depth_mul is not None:
            cfg.model.depth_multiple = depth_mul

        self.backbone, self.neck, channel_counts = build_network(cfg)

        self.necks = nn.ModuleList()
        self.necks.append(self.neck)
        head_config['ch'] = channel_counts

        if head_cls != Detect:
            head_config.pop('anchors')
        self.detection = head_cls(**head_config)

        self._init_head()

        initialize_weights(self)
        self._is_fused = False

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
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.info()
        self._is_fused = True
        return self
