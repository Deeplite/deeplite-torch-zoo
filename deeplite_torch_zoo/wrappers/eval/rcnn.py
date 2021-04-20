import torch
import os.path as osp
from pathlib import Path

import mmcv
from mmcv import Config

from deeplite_torch_zoo.src.poseestimation.evaluation.gpu_test import inference
from deeplite_torch_zoo.src.objectdetection.eval.coco.mask_rcnn import RCNNCOCOEvaluator


__all__ = ["rcnn_eval_coco", "keypoint_rcnn_eval_coco"]


_cfg = {
    "coco_384x288": "deeplite_torch_zoo/src/poseestimation/configs/coco_384x288.py",
}


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def rcnn_eval_coco(model, data_loader, gt=None, device="cuda", net="rcnn"):
    model.to(device)
    with torch.no_grad():
        return RCNNCOCOEvaluator(
            model,
            data_loader.dataset,
            gt=gt,
            net=net
        ).evaluate()


def get_cfg_path(_set="coco_384x288"):
    return str(get_project_root() / _cfg[_set])


def get_cfg(_set="coco_384x288"):
    cfg_path = get_cfg_path(_set=_set)
    cfg = Config.fromfile(cfg_path)
    return cfg


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def keypoint_rcnn_eval_coco(model, data_loader, _set="coco_384x288"):
    cfg = get_cfg(_set=_set)
    dataset = data_loader.dataset
    outputs = inference(model, data_loader)

    eval_config = cfg.get("eval_config", {})
    eval_config = merge_configs(eval_config, dict(metric="mAP"))
    work_dir = "./work_dirs/keypoint_rcnn_{_set}/".format(_set=_set)
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    res = dataset.evaluate(outputs, work_dir, **eval_config)
    return res
