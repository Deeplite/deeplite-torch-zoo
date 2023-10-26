# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner


def patched_get_model(obj, weights=None, cfg=None):
    return weights


def patched_loss_init(obj, model):  # model must be de-paralleled
    device = next(model.parameters()).device  # get model device
    h = model.args  # hyperparameters

    m = model.detection if hasattr(model, 'detection') else model.model[-1]  # Detect() module
    obj.bce = nn.BCEWithLogitsLoss(reduction='none')
    obj.hyp = h
    obj.stride = m.stride  # model strides
    obj.nc = m.nc  # number of classes
    obj.no = m.no
    obj.reg_max = m.reg_max
    obj.device = device

    obj.use_dfl = m.reg_max > 1

    obj.assigner = TaskAlignedAssigner(topk=10, num_classes=obj.nc, alpha=0.5, beta=6.0)
    obj.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=obj.use_dfl).to(device)
    obj.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)


DetectionTrainer.get_model = patched_get_model
v8DetectionLoss.__init__ = patched_loss_init
