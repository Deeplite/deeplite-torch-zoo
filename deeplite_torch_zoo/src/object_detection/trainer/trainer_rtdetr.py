# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch.nn as nn

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.loss import FocalLoss, VarifocalLoss

from deeplite_torch_zoo.utils.ops import HungarianMatcher

def patched_get_model(obj, weights=None, cfg=None):
    return weights

def patched_loss_init(obj, model, 
                      loss_gain=None, 
                      aux_loss=True, 
                      use_fl=True,
                      use_vfl=False,
                      use_uni_match=False,
                      uni_match_ind=0):  # model must be de-paralleled
    
    nn.Module.__init__(obj)
    if loss_gain is None:
            loss_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1, 'mask': 1, 'dice': 1}
    m = model.detection if hasattr(model, 'detection') else model.model[-1]  # Detect() module
    obj.nc = m.nc  # number of classes
    obj.matcher = HungarianMatcher(cost_gain={'class': 2, 'bbox': 5, 'giou': 2})
    obj.loss_gain = loss_gain
    obj.aux_loss = aux_loss
    obj.fl = FocalLoss() if use_fl else None
    obj.vfl = VarifocalLoss() if use_vfl else None

    obj.use_uni_match = use_uni_match
    obj.uni_match_ind = uni_match_ind
    obj.device = next(model.parameters()).device  # get model device

RTDETRTrainer.get_model = patched_get_model
RTDETRDetectionLoss.__init__ = patched_loss_init
