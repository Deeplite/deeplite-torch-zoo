import math

import numpy as np
import torch
import torch.nn as nn

import deeplite_torch_zoo.src.objectdetection.configs.hyp_config as hyp_cfg
from deeplite_torch_zoo.src.objectdetection.yolov5.utils.general import xyxy2cxcywh


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(
    eps=0.1,
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(
            reduction="none"
        )  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class YoloV5Loss(nn.Module):
    def __init__(self, model, num_classes=80, device="cuda"):
        super(YoloV5Loss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.model = model
        self.hyp_params = hyp_cfg.TRAIN  # hyperparameters

    def forward(
        self, p, p_d, raw_targets, labels_length, img_size
    ):  # predictions, targets
        targets = torch.zeros(
            (int(sum(labels_length)), 6), device=self.device, dtype=torch.float32
        )
        cum_index = np.cumsum(labels_length)
        prv_index = 0
        for i, target in enumerate(raw_targets):
            current_index = int(cum_index[i])
            current_length = int(labels_length[i])
            targets[prv_index:current_index, 0] = i
            targets[prv_index:current_index, 1] = target[:current_length, 4]
            targets[prv_index:current_index, 2:] = xyxy2cxcywh(
                target[:current_length, :4]
            )  # convert from x1, y1, x2, y2 to cx, cy, w, h
            prv_index = cum_index[i]

        targets[:, 2:6] /= img_size  # Normalize to 0 - 1

        lcls, lbox, lobj = (
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
        )
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([self.hyp_params["cls_pw"]])
        ).to(self.device)
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([self.hyp_params["obj_pw"]])
        ).to(self.device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        # Focal loss
        g = self.hyp_params["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Losses
        nt = 0  # number of targets
        num_outputs = len(p)  # number of outputs
        balance = (
            [4.0, 1.0, 0.4] if num_outputs == 3 else [4.0, 1.0, 0.4, 0.1]
        )  # P3-5 or P3-6
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

            tobj = torch.zeros_like(pi[..., 0], device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(self.device)  # predicted box
                giou = bbox_iou(
                    pbox.T, tbox[i], x1y1x2y2=False, CIoU=True
                )  # giou(prediction, target)
                lbox += (1.0 - giou).mean()  # giou loss

                # Objectness
                tobj[b, a, gj, gi] = (
                    1.0 - self.hyp_params["giou_loss_ratio"]
                ) + self.hyp_params["giou_loss_ratio"] * giou.detach().clamp(0).type(
                    tobj.dtype
                )  # giou ratio
                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        s = 3 / num_outputs  # output count scaling
        lbox *= self.hyp_params["giou"] * s
        lobj *= self.hyp_params["obj"] * s * (1.4 if num_outputs == 4 else 1.0)
        lcls *= self.hyp_params["cls"] * s
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return (
            loss * bs,
            lbox,
            lobj,
            lcls,
        )  # torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        """
        na is number of anchors - > 3

        """
        # targets = targets[]
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        det = self.model.model[-1]
        na, nt = det.na, targets.shape[0]  # number of anchors, targets
        nl = det.nl
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
            anchors = det.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = (
                    torch.max(r, 1.0 / r).max(2)[0] < self.hyp_params["anchor_t"]
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


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
