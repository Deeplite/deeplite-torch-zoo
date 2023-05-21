# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import time

import numpy as np
import torch
from tqdm import tqdm

from deeplite_torch_zoo.src.object_detection.eval.mean_average_precision import (
    MetricBuilder,
)
from deeplite_torch_zoo.src.object_detection.eval.yolov5_eval.utils import (
    box_iou,
    check_version,
    non_max_suppression,
)
from deeplite_torch_zoo.src.object_detection.eval.yolov5_eval.v8_nms import (
    non_max_suppression as non_max_suppression_v8,
)
from deeplite_torch_zoo.utils import LOGGER


def smart_inference_mode(torch_1_9=check_version(torch.__version__, '1.9.0')):
    # Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator
    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where(
            (iou >= iouv[i]) & correct_class
        )  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def evaluate(
    model,
    dataloader,
    num_classes=None,
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.5,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    half=False,  # use FP16 half-precision inference
    eval_style='coco',
    map_iou_thresh=0.5,
    v8_eval=False,
):
    if v8_eval:
        map_iou_thresh = np.arange(0.5, 1.0, 0.05)

    if num_classes is None:
        num_classes = dataloader.dataset.num_classes

    # Initialize/load model and set device
    device = next(model.parameters()).device  # get model device, PyTorch model
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure
    model.eval()
    cuda = device.type != 'cpu'

    names = (
        model.names if hasattr(model, 'names') else model.module.names
    )  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    s = ('%22s' + '%11s' * 6) % (
        'Class',
        'Images',
        'Instances',
        'P',
        'R',
        'mAP50',
        'mAP50-95',
    )
    pbar = tqdm(
        dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
    )  # progress bar

    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d", async_mode=False, num_classes=num_classes
    )

    LOGGER.info('Inference on test set')
    for im, targets, _, shapes in pbar:
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        # im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        preds = model(im, augment=augment)

        # NMS
        if not v8_eval:
            preds = non_max_suppression(
                preds,
                conf_thres,
                iou_thres,
                labels=[],
                multi_label=True,
                agnostic=single_cls,
                max_det=max_det,
            )
        else:
            preds = non_max_suppression_v8(
                preds,
                conf_thres,
                iou_thres,
                labels=[],
                multi_label=True,
                agnostic=single_cls,
                max_det=max_det,
            )

        for i, pred in enumerate(preds):
            orig_shape = tuple(shapes[i].numpy())
            pred = pred.cpu().numpy()

            pred = scale_predictions(
                pred, height, orig_shape, (0, np.inf), conf_thresh=conf_thres
            )

            p = np.zeros(pred.shape)
            p[:, :4] = pred[:, :4]
            p[:, 4] = pred[:, 5]
            p[:, 5] = pred[:, 4]
            jj = 0
            for j in range(targets[i, ...].shape[0]):
                if any(targets[i, j, :]):
                    jj += 1
            gt = np.zeros((jj, 7))
            gt[:, :5] = targets[i, :jj, :].cpu().numpy()

            gt_coor = gt[:, :4]
            org_h, org_w = orig_shape
            resize_ratio = min(1.0 * height / org_w, 1.0 * height / org_h)
            dw = (height - resize_ratio * org_w) / 2
            dh = (height - resize_ratio * org_h) / 2

            gt_coor[:, 0::2] = 1.0 * (gt_coor[:, 0::2] - dw) / resize_ratio
            gt_coor[:, 1::2] = 1.0 * (gt_coor[:, 1::2] - dh) / resize_ratio

            metric_fn.add(p, gt)

    LOGGER.info('Computing mAP value')
    t1 = time.perf_counter()
    if eval_style == 'coco':
        metrics = metric_fn.value(
            iou_thresholds=map_iou_thresh,
            recall_thresholds=np.arange(0.0, 1.01, 0.01),
            mpolicy='soft',
        )
    elif eval_style == 'voc':
        metrics = metric_fn.value(iou_thresholds=map_iou_thresh)
    t2 = time.perf_counter()
    LOGGER.info(f'Finished in {t2 - t1} sec')

    APs = {'mAP': metrics['mAP']}
    LOGGER.info(APs)

    if isinstance(map_iou_thresh, float):
        for cls_id, ap_dict in metrics[map_iou_thresh].items():
            APs[cls_id] = ap_dict['ap']
    return APs


def scale_predictions(
    pred_bbox, test_input_size, org_img_shape, valid_scale, conf_thresh
):
    """
    The prediction frame is filtered to remove frames with unreasonable scales
    """
    pred_coor = pred_bbox[:, :4]
    scores = pred_bbox[:, 4]
    classes = pred_bbox[:, 5]

    # (1)
    # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
    # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
    # It should be noted that no matter what data augmentation method we use during training, it does not affect the transformation method here
    # Suppose we use conversion method A for the input test image, then the conversion method for bbox here is the reverse process of method A
    org_h, org_w = org_img_shape
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    dw = (test_input_size - resize_ratio * org_w) / 2
    dh = (test_input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # (2) Cut off the part of the predicted bbox beyond the original image
    pred_coor = np.concatenate(
        [
            np.maximum(pred_coor[:, :2], [0, 0]),
            np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
        ],
        axis=-1,
    )
    # (3) Set the coor of the invalid bbox to 0
    invalid_mask = np.logical_or(
        (pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3])
    )
    pred_coor[invalid_mask] = 0

    # (4) Remove bboxes that are not within the valid range
    bboxes_scale = np.sqrt(
        np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1)
    )
    scale_mask = np.logical_and(
        (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1])
    )

    # (5) Remove the bbox whose score is lower than score_threshold
    score_mask = scores > conf_thresh

    mask = np.logical_and(scale_mask, score_mask)

    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]

    bboxes = np.concatenate(
        [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1
    )

    return bboxes
