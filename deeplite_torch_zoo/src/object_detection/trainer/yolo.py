# Ultralytics YOLO 🚀, AGPL-3.0 license

import types
from copy import deepcopy

import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, callbacks
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.models.yolo.model import YOLO

from deeplite_torch_zoo import get_model
from deeplite_torch_zoo.src.object_detection.yolo.config_parser import HEAD_NAME_MAP


def patched_init(obj, model_name=None, torch_model=None, num_classes=None,
                 task=None, session=None, pretrained=False, pretraining_dataset='coco'):
    if isinstance(obj, nn.Module):
        nn.Module.__init__(obj)
    if model_name is None and torch_model is None:
        raise ValueError('Either a `model_name` string or a `torch_model` (nn.Module object) must be passed '
                         'to instantiate a trainer object.')
    obj.callbacks = callbacks.get_default_callbacks()
    obj.predictor = None  # reuse predictor
    obj.model = None  # model object
    obj.trainer = None  # trainer object
    obj.task = task if task is not None else 'detect' # task type
    obj.ckpt = True  # if loaded from *.pt
    obj.cfg = None  # if loaded from *.yaml
    obj.ckpt_path = None
    obj.overrides = {}  # overrides for trainer object
    obj.metrics = None  # validation/training metrics
    obj.session = session  # HUB session
    obj.num_classes = num_classes

    obj.cfg = model_name
    if model_name == 'yolov8n.pt':
        model_name = 'yolo8n'
    if model_name is not None:
        obj.model = get_model(
            model_name=model_name,
            dataset_name=pretraining_dataset,
            pretrained=pretrained,
            num_classes=num_classes,
            custom_head='yolo8',  # can only work with v8 head as of now
        )
    else:
        obj.model = torch_model

    obj.model._forward = obj.model.forward

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self._forward(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = v8DetectionLoss(self)
        preds = self._forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    obj.model.loss = types.MethodType(loss, obj.model)
    obj.model.forward = types.MethodType(forward, obj.model)

    obj.model.names = [''] if not num_classes else [f'class{i}' for i in range(num_classes)]
    obj.overrides['model'] = obj.cfg

    # Below added to allow export from yamls
    args = {**DEFAULT_CFG_DICT, **obj.overrides}  # combine model and default args, preferring model args
    obj.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    obj.model.task = obj.task
    obj.model.model_name = model_name


def patched_export(obj, model_name='model', **kwargs):
    obj.model.yaml = {'yaml_file': model_name}

    obj._check_is_pytorch_model()
    overrides = obj.overrides.copy()
    overrides.update(kwargs)
    overrides['mode'] = 'export'
    args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
    args.task = obj.task
    if args.imgsz == DEFAULT_CFG.imgsz:
        args.imgsz = obj.model.args['imgsz']  # use trained imgsz unless custom value is passed
    if args.batch == DEFAULT_CFG.batch:
        args.batch = 1  # default to 1 if not modified

    # Update model
    model = deepcopy(obj.model).to(obj.device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, tuple(HEAD_NAME_MAP.values())):
            m.dynamic = args.dynamic
            m.export = True
            m.format = args.format

    model.yaml_file = model_name
    return Exporter(overrides=args, _callbacks=obj.callbacks)(model=model)


YOLO.__init__ = patched_init
YOLO.export = patched_export
