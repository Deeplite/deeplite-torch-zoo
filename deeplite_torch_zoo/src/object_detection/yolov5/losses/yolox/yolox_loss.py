# Source: https://github.com/iscyy/yoloair

import torch

from deeplite_torch_zoo.src.object_detection.yolov5.losses.loss_utils import (
    get_yolov5_targets,
    is_parallel,
)


class ComputeXLoss:
    # Compute losses
    def __init__(self, model, device="cuda", autobalance=False):
        super(ComputeXLoss, self).__init__()

        det = (
            model.module.model[-1] if is_parallel(model) else model.model[-1]
        )  # Detect() module
        self.det = det
        self.device = device

    def __call__(
        self, p, raw_targets, labels_length, img_size
    ):  # predictions, targets, model
        targets = get_yolov5_targets(raw_targets, labels_length, img_size, self.device)

        if (not self.det.training) or (len(p) == 0) or (len(targets) == 0):
            return torch.tensor(
                0.0, device=self.device, requires_grad=True
            ), torch.zeros(4, device=self.device, requires_grad=True)

        loss, iou_loss, obj_loss, cls_loss, l1_loss, _ = self.det.get_losses(
            *p, targets, dtype=p[0].dtype
        )

        return loss, torch.hstack((iou_loss, obj_loss, cls_loss, l1_loss)).detach()
