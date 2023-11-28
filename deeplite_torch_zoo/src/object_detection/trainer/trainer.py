# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import numpy as np

from ultralytics.yolo.v8.detect.train import DetectionTrainer, Loss, make_anchors
from ultralytics.yolo.utils.tal import TaskAlignedAssigner
from ultralytics.yolo.utils.loss import BboxLoss
import ultralytics.yolo.engine.trainer

from deeplite_torch_zoo.src.object_detection.trainer.yolo import YOLO
from deeplite_torch_zoo.utils import LOGGER, colorstr


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


def patched_check_amp(model):
    device = next(model.parameters()).device  # get model device
    if device.type in ('cpu', 'mps'):
        return False  # AMP only used on CUDA devices

    def amp_allclose(m, im):
        # All close FP32 vs AMP results
        a = m(im, device=device, verbose=False)[0].boxes.boxes  # FP32 inference
        with torch.cuda.amp.autocast(True):
            b = m(im, device=device, verbose=False)[0].boxes.boxes  # AMP inference
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

    im = np.ones((640, 640, 3))
    prefix = colorstr('AMP: ')
    LOGGER.info(f'{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...')
    model = YOLO(model_name='yolo8n')
    try:
        assert amp_allclose(model, im)
        LOGGER.info(f'{prefix}checks passed ✅')
    except AssertionError:
        LOGGER.warning(f'{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to '
                       f'NaN losses or zero-mAP results, so AMP will be disabled during training.')
        return False
    return True


def patched_collate_fn(obj, batch):
    """Collates data samples into batches."""
    new_batch = {}
    keys = batch[0].keys()
    values = list(zip(*[list(b.values()) for b in batch]))
    for i, k in enumerate(keys):
        value = values[i]
        if k == 'img':
            value = torch.stack(value, 0)
        if k in ['masks', 'keypoints', 'bboxes', 'cls']:
            value = torch.cat(value, 0)
        new_batch[k] = value
    new_batch['batch_idx'] = list(new_batch['batch_idx'])
    for i in range(len(new_batch['batch_idx'])):
        new_batch['batch_idx'][i] += i  # add target image index for build_targets()
    new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)

    # skipping concatenation of targets in new_batch because yolov8 Loss does this
    shapes = None
    if 'ratio_pad' in new_batch:
        shapes = [(ori_shape, ratio_pad) for ori_shape, ratio_pad
                in zip(new_batch['ori_shape'], new_batch['ratio_pad'])]
    new_batch['shapes'] = shapes
    return new_batch


class V8UltralyticsLoss(Loss):
    def __call__(self, pred, batch):
        batch = {
            'batch_idx': batch[:, 0],
            'cls': batch[:, 1],
            'bboxes': batch[:, 2:],
        }
        return super().__call__(pred, batch)  # ultralytics takes batch dict as targets instead of tensor


DetectionTrainer.get_model = patched_get_model
Loss.__init__ = patched_loss_init
ultralytics.yolo.engine.trainer.check_amp = patched_check_amp
