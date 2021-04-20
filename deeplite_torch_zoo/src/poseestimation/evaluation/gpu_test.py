import os.path as osp
import pickle
import shutil
import tempfile
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from deeplite_torch_zoo.src.poseestimation.datasets.post_processing import transform_preds



def xywh_cs(bboxes, dataset):
    x = bboxes[:, 0]
    y = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    cs = []
    for _x, _y, _w, _h in zip(x, y, w, h):
        c, s = dataset._xywh2cs(_x, _y, _w, _h)
        cs.append(list(c) + list(s))
    cs = np.array(cs)
    bboxes[:, :4] = cs
    return bboxes


def remap_results(result, img_metas, img_size, dataset):
    result = result[0]
    boxes = result['boxes'].cpu().numpy()
    labels = result['labels'].cpu().numpy()
    scores = result['scores'].cpu().numpy()
    keypoints = result['keypoints'].cpu().numpy()
    keypoints_scores = result['keypoints_scores'].cpu().numpy()
    keypoints[:, :, 2] = keypoints_scores

    bboxes = np.zeros((len(boxes), 6))
    bboxes[:, 0] = boxes[:, 0]
    bboxes[:, 1] = boxes[:, 1]
    bboxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    bboxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    bboxes[:, 4] = bboxes[:, 2] * bboxes[:, 3]
    bboxes[:, 5] = scores
    return keypoints, xywh_cs(bboxes, dataset), img_metas['image_file'], None


def inference(model, data_loader, device="cuda"):
    """Generates keypoints results for a given model witn a given dataloader.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """

    model.eval()
    model = model.to(device)
    results = []
    dataset = data_loader.dataset
    count = 0
    for data in data_loader:
        with torch.no_grad():
            try:
                result = model(data['img'].to(device))
            except Exception as e:
                count = count + 1
                continue
            result = remap_results(result, data['img_metas'].data[0][0], data['img'].shape[-2:], dataset)

        results.append(result)

    print(f"num. of skipped {count}")
    return results
