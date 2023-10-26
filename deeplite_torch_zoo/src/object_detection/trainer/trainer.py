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
    # print('patched')
    # targets = torch.cat([  # yolov8 loss performs concatenation in loss function
    #     new_batch['batch_idx'].unsqueeze(-1),
    #     new_batch['cls'],
    #     new_batch['bboxes']
    # ], axis=1)

# torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)

    shapes = None
    if 'ratio_pad' in new_batch:
        shapes = [(ori_shape, ratio_pad) for ori_shape, ratio_pad
                in zip(new_batch['ori_shape'], new_batch['ratio_pad'])]
    new_batch['shapes'] = shapes
    return new_batch


def patched_yolo_loss(obj, preds, batch):
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    loss = torch.zeros(3, device=obj.device)  # box, cls, dfl
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], obj.no, -1) for xi in feats], 2).split(
        (obj.reg_max * 4, obj.nc), 1)

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=obj.device, dtype=dtype) * obj.stride[0]  # image size (h,w)
    anchor_points, stride_tensor = make_anchors(feats, obj.stride, 0.5)

    # Patch targets because zoo dataloader already concatenated them
    # targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
    targets = batch
    targets = obj.preprocess(targets.to(obj.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

    # pboxes
    pred_bboxes = obj.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

    _, target_bboxes, target_scores, fg_mask, _ = obj.assigner(
        pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

    target_scores_sum = max(target_scores.sum(), 1)

    # cls loss
    # loss[1] = obj.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
    loss[1] = obj.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

    # bbox loss
    if fg_mask.sum():
        target_bboxes /= stride_tensor
        loss[0], loss[2] = obj.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                            target_scores_sum, fg_mask)

    loss[0] *= obj.hyp.box  # box gain
    loss[1] *= obj.hyp.cls  # cls gain
    loss[2] *= obj.hyp.dfl  # dfl gain

    return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


DetectionTrainer.get_model = patched_get_model
Loss.__init__ = patched_loss_init
ultralytics.yolo.engine.trainer.check_amp = patched_check_amp
