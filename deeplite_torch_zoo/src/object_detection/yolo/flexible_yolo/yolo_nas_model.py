import torch.nn as nn

from deeplite_torch_zoo.utils import initialize_weights

from deeplite_torch_zoo.src.object_detection.yolo.heads import Detect
from deeplite_torch_zoo.src.object_detection.yolo.config_parser import HEAD_NAME_MAP
from deeplite_torch_zoo.src.object_detection.yolo.anchors import ANCHOR_REGISTRY

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.model import FlexibleYOLO


class YOLONAS(FlexibleYOLO):
    def __init__(
        self,
        nc=80,
        arch_name='yolo_nas_l_arch_params',
        backbone_cfg=None,
        neck_cfg=None,
        anchors=None,
        custom_head=None,
        in_channels=3,
        neck_depth_factor=None,
        neck_width_factor=None,
    ):
        import super_gradients.common.factories.detection_modules_factory as det_factory
        from super_gradients.training.models.arch_params_factory import get_arch_params

        nn.Module.__init__(self)
        self.yaml = None

        head_config = {
            'nc': nc,
            'anchors': anchors if anchors is not None \
                else ANCHOR_REGISTRY.get('default')(),
        }

        default_arch_params = get_arch_params(arch_name)

        if backbone_cfg is None:
            backbone_cfg = default_arch_params['backbone']
        if neck_cfg is None:
            neck_cfg = default_arch_params['neck']

        if neck_depth_factor is not None:
            for neck in neck_cfg.values():
                for stage in neck.values():
                    for block in stage.values():
                        block['depth_mult'] = neck_depth_factor

        if neck_width_factor is not None:
            for neck in neck_cfg.values():
                for stage in neck.values():
                    for block in stage.values():
                        block['width_mult'] = neck_depth_factor

        factory = det_factory.DetectionModulesFactory()

        head_cls = Detect
        if custom_head is not None:
            if custom_head not in HEAD_NAME_MAP:
                raise ValueError(f'Incorrect YOLO head name {custom_head}. Choices: {list(HEAD_NAME_MAP.keys())}')
            head_cls = HEAD_NAME_MAP[custom_head]

        self.backbone = factory.get(factory.insert_module_param(backbone_cfg, 'in_channels', in_channels))
        ch_in = self.backbone.out_channels

        self.necks = nn.ModuleList()
        neck = factory.get(factory.insert_module_param(neck_cfg, 'in_channels', ch_in))
        ch_in = neck.out_channels
        self.necks.append(neck)
        head_config['ch'] = ch_in

        if head_cls != Detect:
            head_config.pop('anchors')
        self.detection = head_cls(**head_config)

        self._init_head()

        initialize_weights(self)
        self._is_fused = False
