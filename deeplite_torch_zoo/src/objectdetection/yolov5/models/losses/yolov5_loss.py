# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# The file is modified by Deeplite Inc. from the original implementation on Feb 24, 2023
# Make it more general for different yolo versions
# Add Compatibility to our engine

from copy import copy

import torch
import torch.nn as nn

import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_default as hyp_cfg_default
from deeplite_torch_zoo.src.objectdetection.yolov5.models.losses.loss_utils import (
    FocalLoss, bbox_iou, de_parallel, get_yolov5_targets, smooth_BCE)


class YoloV5Loss(nn.Module):
    def __init__(self, model, num_classes=80, device="cuda", autobalance=False, hyp_cfg=None):
        super(YoloV5Loss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.model = model
        if hyp_cfg is None:
            hyp_cfg = hyp_cfg_default
        self.hyp = hyp_cfg.TRAIN  # hyperparameters
        self.sort_obj_iou = False

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=self.hyp.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = self.hyp['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        if hasattr(model, 'model'):
            det = de_parallel(model).model[-1]  # Detect() module
        if hasattr(model, 'detection'):
            det = de_parallel(model).detection

        self.na = copy(det.na)
        self.nl = copy(det.nl)
        self.nc = copy(det.nc)
        self.anchors = copy(det.anchors).to(device)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance

    def forward(
        self, p, raw_targets, labels_length, img_size
    ):  # predictions, targets
        targets = get_yolov5_targets(raw_targets, labels_length, img_size, self.device)

        lcls, lbox, lobj = (
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
        )
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['giou']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return (
            loss * bs,
            torch.tensor([lbox, lobj, lcls], requires_grad=True).to(self.device)
        )

    def build_targets(self, p, targets):
        """
        na is number of anchors - > 3

        """
        # targets = targets[]
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        nl = self.nl
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = (
            torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[:, :, None]), 2
        )  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = (
                    torch.max(r, 1.0 / r).max(2)[0] < self.hyp["anchor_t"]
                )  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
