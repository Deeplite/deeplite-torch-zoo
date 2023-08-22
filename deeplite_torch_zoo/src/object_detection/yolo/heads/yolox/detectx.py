"""
YOLOX-specific modules
Source: https://github.com/iscyy/yoloair
"""

import math

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from deeplite_torch_zoo.src.object_detection.yolo.common import *
from deeplite_torch_zoo.src.object_detection.yolo.experimental import *
from deeplite_torch_zoo.src.object_detection.yolo.losses.yolox.yolox_loss import *
from deeplite_torch_zoo.utils import LOGGER


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target, xyxy=False):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        if xyxy:
            tl = torch.max(pred[:, :2], target[:, :2])
            br = torch.min(pred[:, 2:], target[:, 2:])
            area_p = torch.prod(pred[:, 2:] - pred[:, :2], 1)
            area_g = torch.prod(target[:, 2:] - target[:, :2], 1)
        else:
            tl = torch.max(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            br = torch.min(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_p = torch.prod(pred[:, 2:], 1)
            area_g = torch.prod(target[:, 2:], 1)

        hw = (br - tl).clamp(min=0)  # [rows, 2]
        area_i = torch.prod(hw, 1)

        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            if xyxy:
                c_tl = torch.min(pred[:, :2], target[:, :2])
                c_br = torch.max(pred[:, 2:], target[:, 2:])
            else:
                c_tl = torch.min(
                    (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
                )
                c_br = torch.max(
                    (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
                )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class DetectX(nn.Module):
    stride = [8, 16, 32]
    onnx_dynamic = False  # ONNX export parameter
    export = False

    def __init__(
        self,
        num_classes,
        anchors=1,
        in_channels=(128, 128, 128, 128, 128, 128),
        inplace=True,
        prior_prob=1e-2,
    ):
        super().__init__()
        if isinstance(anchors, (list, tuple)):
            self.n_anchors = len(anchors)
        else:
            self.n_anchors = anchors
        self.num_classes = num_classes

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        cls_in_channels = in_channels[0::2]
        reg_in_channels = in_channels[1::2]
        for cls_in_channel, reg_in_channel in zip(cls_in_channels, reg_in_channels):
            cls_pred = nn.Conv2d(
                in_channels=cls_in_channel,
                out_channels=self.n_anchors * self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            reg_pred = nn.Conv2d(
                in_channels=reg_in_channel,
                out_channels=4,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            obj_pred = nn.Conv2d(
                in_channels=reg_in_channel,
                out_channels=self.n_anchors * 1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.cls_preds.append(cls_pred)
            self.reg_preds.append(reg_pred)
            self.obj_preds.append(obj_pred)

        self.nc = self.num_classes
        self.nl = len(cls_in_channels)
        self.na = self.n_anchors

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.xy_shifts = [torch.zeros(1)] * len(in_channels)
        self.org_grids = [torch.zeros(1)] * len(in_channels)
        self.grid_sizes = [[0, 0, 0] for _ in range(len(in_channels))]
        self.expanded_strides = [torch.zeros(1)] * len(in_channels)
        self.center_ltrbes = [torch.zeros(1)] * len(in_channels)
        self.center_radius = 2.5

        self.prior_prob = prior_prob
        self.inplace = inplace

    def initialize_biases(self):
        prior_prob = self.prior_prob
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _forward(self, xin):
        outputs = []
        cls_preds = []
        bbox_preds = []
        obj_preds = []
        origin_preds = []
        org_xy_shifts = []
        xy_shifts = []
        expanded_strides = []
        center_ltrbes = []

        cls_xs = xin[0::2]
        reg_xs = xin[1::2]
        in_type = xin[0].type()
        h, w = reg_xs[0].shape[2:4]
        h *= self.stride[0]
        w *= self.stride[0]
        for k, (stride_this_level, cls_x, reg_x) in enumerate(
            zip(self.stride, cls_xs, reg_xs)
        ):
            cls_output = self.cls_preds[k](cls_x)
            reg_output = self.reg_preds[k](reg_x)
            obj_output = self.obj_preds[k](reg_x)

            if self.training:
                batch_size = cls_output.shape[0]
                hsize, wsize = cls_output.shape[-2:]
                size = hsize * wsize
                cls_output = (
                    cls_output.view(batch_size, -1, size).permute(0, 2, 1).contiguous()
                )
                reg_output = (
                    reg_output.view(batch_size, 4, size).permute(0, 2, 1).contiguous()
                )
                obj_output = (
                    obj_output.view(batch_size, 1, size).permute(0, 2, 1).contiguous()
                )
                if self.use_l1:
                    origin_preds.append(reg_output.clone())
                (
                    output,
                    grid,
                    xy_shift,
                    expanded_stride,
                    center_ltrb,
                ) = self.get_output_and_grid(
                    reg_output, hsize, wsize, k, stride_this_level, in_type
                )

                org_xy_shifts.append(grid)
                xy_shifts.append(xy_shift)
                expanded_strides.append(expanded_stride)
                center_ltrbes.append(center_ltrb)
                cls_preds.append(cls_output)
                bbox_preds.append(output)
                obj_preds.append(obj_output)
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
                outputs.append(output)

        if self.training:
            bbox_preds = torch.cat(bbox_preds, 1)
            obj_preds = torch.cat(obj_preds, 1)
            cls_preds = torch.cat(cls_preds, 1)

            org_xy_shifts = torch.cat(org_xy_shifts, 1)
            xy_shifts = torch.cat(xy_shifts, 1)
            expanded_strides = torch.cat(expanded_strides, 1)
            center_ltrbes = torch.cat(center_ltrbes, 1)

            if self.use_l1:
                origin_preds = torch.cat(origin_preds, 1)
            else:
                origin_preds = bbox_preds.new_zeros(1)

            whwh = torch.Tensor([[w, h, w, h]]).type_as(bbox_preds)
            return (
                bbox_preds,
                cls_preds,
                obj_preds,
                origin_preds,
                org_xy_shifts,
                xy_shifts,
                expanded_strides,
                center_ltrbes,
                whwh,
            )
        else:
            return outputs

    def forward(self, x):
        outputs = self._forward(x)

        if self.training:
            return outputs
        else:
            if self.export:
                # self.hw = [torch.tensor([80, 80]),
                #            torch.tensor([40, 40]),
                #            torch.tensor([20, 20])]
                bs = -1
                no = outputs[0].shape[1]
            else:
                bs, no = outputs[0].shape[0], outputs[0].shape[1]

            self.hw = [out.shape[-2:] for out in outputs]

            # [batch, n_anchors_all, 85]
            outs = []
            for i in range(len(outputs)):
                h, w = self.hw[i]
                out = outputs[i]
                outs.append(out.view(bs, no, h * w))
            out = torch.cat(outs, dim=-1).permute(0, 2, 1)
            out = self.decode_outputs(out, dtype=x[0].type())
            return (out,)

    def forward_export(self, x):
        cls_xs = x[0::2]
        reg_xs = x[1::2]
        outputs = []
        for k, (stride_this_level, cls_x, reg_x) in enumerate(
            zip(self.stride, cls_xs, reg_xs)
        ):
            cls_output = self.cls_preds[k](cls_x)
            reg_output = self.reg_preds[k](reg_x)
            obj_output = self.obj_preds[k](reg_x)
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            outputs.append(output)
        outputs = torch.cat(
            [out.flatten(start_dim=2) for out in outputs], dim=2
        ).permute(0, 2, 1)
        return outputs

    def get_output_and_grid(self, reg_box, hsize, wsize, k, stride, dtype):
        grid_size = self.grid_sizes[k]
        if (
            (grid_size[0] != hsize)
            or (grid_size[1] != wsize)
            or (grid_size[2] != stride)
        ):
            grid_size[0] = hsize
            grid_size[1] = wsize
            grid_size[2] = stride

            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = (
                torch.stack((xv, yv), 2).view(1, -1, 2).type(dtype).contiguous()
            )  # [1, 1*hsize*wsize, 2]
            self.grids[k] = grid
            xy_shift = (grid + 0.5) * stride
            self.xy_shifts[k] = xy_shift
            expanded_stride = torch.full(
                (1, grid.shape[1], 1), stride, dtype=grid.dtype, device=grid.device
            )
            self.expanded_strides[k] = expanded_stride
            center_radius = self.center_radius * expanded_stride
            center_radius = center_radius.expand_as(xy_shift)
            center_lt = center_radius + xy_shift
            center_rb = center_radius - xy_shift
            center_ltrb = torch.cat([center_lt, center_rb], dim=-1)
            self.center_ltrbes[k] = center_ltrb

        xy_shift = self.xy_shifts[k]
        grid = self.grids[k]
        expanded_stride = self.expanded_strides[k]
        center_ltrb = self.center_ltrbes[k]

        # l, t, r, b
        half_wh = torch.exp(reg_box[..., 2:4]) * (stride / 2)
        reg_box[..., :2] = (reg_box[..., :2] + grid) * stride
        reg_box[..., 2:4] = reg_box[..., :2] + half_wh
        reg_box[..., :2] = reg_box[..., :2] - half_wh

        return reg_box, grid, xy_shift, expanded_stride, center_ltrb

    def make_grid(self, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.stride):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        return grids, strides

    def decode_outputs(self, outputs, dtype):
        grids, strides = self.make_grid(dtype)
        xy = (outputs[..., :2] + grids) * strides
        wh = torch.exp(outputs[..., 2:4]) * strides
        score = outputs[..., 4:]
        out = torch.cat([xy, wh, score], -1)
        return out

    def get_losses(
        self,
        bbox_preds,
        cls_preds,
        obj_preds,
        origin_preds,
        org_xy_shifts,
        xy_shifts,
        expanded_strides,
        center_ltrbes,
        whwh,
        labels,
        dtype,
    ):
        # calculate targets
        nlabel = labels[:, 0].long().bincount(minlength=cls_preds.shape[0]).tolist()
        batch_gt_classes = labels[:, 1].type_as(cls_preds).contiguous()  # [num_gt, 1]
        batch_org_gt_bboxes = labels[
            :, 2:6
        ].contiguous()  # [num_gt, 4]  bbox: cx, cy, w, h
        batch_org_gt_bboxes.mul_(whwh)
        batch_gt_bboxes = torch.empty_like(
            batch_org_gt_bboxes
        )  # [num_gt, 4]  bbox: l, t, r, b
        batch_gt_half_wh = batch_org_gt_bboxes[:, 2:] / 2
        batch_gt_bboxes[:, :2] = batch_org_gt_bboxes[:, :2] - batch_gt_half_wh
        batch_gt_bboxes[:, 2:] = batch_org_gt_bboxes[:, :2] + batch_gt_half_wh
        batch_org_gt_bboxes = batch_org_gt_bboxes.type_as(bbox_preds)
        batch_gt_bboxes = batch_gt_bboxes.type_as(bbox_preds)
        del batch_gt_half_wh

        total_num_anchors = bbox_preds.shape[1]

        cls_targets = []
        reg_targets = []
        l1_targets = []
        fg_mask_inds = []

        num_fg = 0.0
        num_gts = 0
        index_offset = 0
        batch_size = bbox_preds.shape[0]
        for batch_idx in range(batch_size):
            num_gt = int(nlabel[batch_idx])
            if num_gt == 0:
                cls_target = bbox_preds.new_zeros((0, self.num_classes))
                reg_target = bbox_preds.new_zeros((0, 4))
                l1_target = bbox_preds.new_zeros((0, 4))
            else:
                _num_gts = num_gts + num_gt
                org_gt_bboxes_per_image = batch_org_gt_bboxes[num_gts:_num_gts]
                gt_bboxes_per_image = batch_gt_bboxes[num_gts:_num_gts]
                # gt_classes = batch_gt_classes[:,num_gts:_num_gts]
                gt_classes = batch_gt_classes[num_gts:_num_gts]
                num_gts = _num_gts
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask_ind,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        num_gt,
                        total_num_anchors,
                        org_gt_bboxes_per_image,
                        gt_bboxes_per_image,
                        gt_classes,
                        self.num_classes,
                        bboxes_preds_per_image,
                        cls_preds_per_image,
                        obj_preds_per_image,
                        center_ltrbes,
                        xy_shifts,
                    )
                except RuntimeError:
                    LOGGER.warning(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    LOGGER.warning("------------CPU Mode for This Batch-------------")
                    _org_gt_bboxes_per_image = org_gt_bboxes_per_image.cpu().float()
                    _gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
                    _bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
                    _cls_preds_per_image = cls_preds_per_image.cpu().float()
                    _obj_preds_per_image = obj_preds_per_image.cpu().float()
                    _gt_classes = gt_classes.cpu().float()
                    _center_ltrbes = center_ltrbes.cpu().float()
                    _xy_shifts = xy_shifts.cpu()

                    (
                        gt_matched_classes,
                        fg_mask_ind,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        num_gt,
                        total_num_anchors,
                        _org_gt_bboxes_per_image,
                        _gt_bboxes_per_image,
                        _gt_classes,
                        self.num_classes,
                        _bboxes_preds_per_image,
                        _cls_preds_per_image,
                        _obj_preds_per_image,
                        _center_ltrbes,
                        _xy_shifts,
                    )

                    gt_matched_classes = gt_matched_classes.cuda()
                    fg_mask_ind = fg_mask_ind.cuda()
                    pred_ious_this_matching = pred_ious_this_matching.cuda()
                    matched_gt_inds = matched_gt_inds.cuda()

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.view(
                    -1, 1
                )  # [num_gt, num_classes]
                reg_target = gt_bboxes_per_image[matched_gt_inds]  # [num_gt, 4]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        bbox_preds.new_empty((num_fg_img, 4)),
                        org_gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask_ind],
                        xy_shifts=org_xy_shifts[0][fg_mask_ind],
                    )
                if index_offset > 0:
                    fg_mask_ind.add_(index_offset)
                fg_mask_inds.append(fg_mask_ind)
            index_offset += total_num_anchors

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        fg_mask_inds = torch.cat(fg_mask_inds, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_mask_inds], reg_targets, True)
        ).sum() / num_fg
        obj_preds = obj_preds.view(-1, 1)
        obj_targets = torch.zeros_like(obj_preds).index_fill_(0, fg_mask_inds, 1)
        loss_obj = (self.bcewithlog_loss(obj_preds, obj_targets)).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_mask_inds], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_mask_inds], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = torch.zeros_like(loss_iou)

        reg_weight = 5.0
        loss_iou = reg_weight * loss_iou
        loss = loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    @staticmethod
    def get_l1_target(l1_target, gt, stride, xy_shifts, eps=1e-8):
        l1_target[:, 0:2] = gt[:, 0:2] / stride - xy_shifts
        l1_target[:, 2:4] = torch.log(gt[:, 2:4] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        total_num_anchors,
        org_gt_bboxes_per_image,
        gt_bboxes_per_image,
        gt_classes,
        num_classes,
        bboxes_preds_per_image,
        cls_preds_per_image,
        obj_preds_per_image,
        center_ltrbes,
        xy_shifts,
    ):
        fg_mask_inds, is_in_boxes_and_center = self.get_in_boxes_info(
            org_gt_bboxes_per_image,
            gt_bboxes_per_image,
            center_ltrbes,
            xy_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask_inds]
        cls_preds_ = cls_preds_per_image[fg_mask_inds]
        obj_preds_ = obj_preds_per_image[fg_mask_inds]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        pair_wise_ious = self.bboxes_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, True, inplace=True
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        cls_preds_ = (
            cls_preds_.float()
            .sigmoid_()
            .unsqueeze(0)
            .expand(num_gt, num_in_boxes_anchor, num_classes)
        )
        obj_preds_ = (
            obj_preds_.float()
            .sigmoid_()
            .unsqueeze(0)
            .expand(num_gt, num_in_boxes_anchor, 1)
        )
        cls_preds_ = (cls_preds_ * obj_preds_).sqrt_()

        del obj_preds_

        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), num_classes).float()
        gt_cls_per_image = gt_cls_per_image[:, None, :].expand(
            num_gt, num_in_boxes_anchor, num_classes
        )

        with autocast(enabled=False):
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_, gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_, gt_cls_per_image

        # 负例给非常大的cost（100000.0及以上）
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )
        del pair_wise_cls_loss, pair_wise_ious_loss, is_in_boxes_and_center

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask_inds,
        ) = self.dynamic_k_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask_inds
        )
        del cost, pair_wise_ious

        return (
            gt_matched_classes,
            fg_mask_inds,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    @staticmethod
    def get_in_boxes_info(
        org_gt_bboxes_per_image,
        gt_bboxes_per_image,
        center_ltrbes,
        xy_shifts,
        total_num_anchors,
        num_gt,
    ):
        xy_centers_per_image = xy_shifts.expand(num_gt, total_num_anchors, 2)
        gt_bboxes_per_image = gt_bboxes_per_image[:, None, :].expand(
            num_gt, total_num_anchors, 4
        )

        b_lt = xy_centers_per_image - gt_bboxes_per_image[..., :2]
        b_rb = gt_bboxes_per_image[..., 2:] - xy_centers_per_image
        bbox_deltas = torch.cat([b_lt, b_rb], 2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        center_ltrbes = center_ltrbes.expand(num_gt, total_num_anchors, 4)
        org_gt_xy_center = org_gt_bboxes_per_image[:, 0:2]
        org_gt_xy_center = torch.cat([-org_gt_xy_center, org_gt_xy_center], dim=-1)
        org_gt_xy_center = org_gt_xy_center[:, None, :].expand(
            num_gt, total_num_anchors, 4
        )
        center_deltas = org_gt_xy_center + center_ltrbes
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return torch.nonzero(is_in_boxes_anchor)[..., 0], is_in_boxes_and_center

    @staticmethod
    def dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask_inds):
        # Dynamic K
        device = cost.device
        matching_matrix = torch.zeros(
            cost.shape, dtype=torch.uint8, device=device
        )  # [num_gt, fg_count]

        ious_in_boxes_matrix = pair_wise_ious  # [num_gt, fg_count]
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = topk_ious.sum(1).int().clamp_min_(1)
        if num_gt > 3:
            min_k, max_k = torch._aminmax(dynamic_ks)
            min_k, max_k = min_k.item(), max_k.item()
            if min_k != max_k:
                offsets = torch.arange(
                    0,
                    matching_matrix.shape[0] * matching_matrix.shape[1],
                    step=matching_matrix.shape[1],
                    dtype=torch.int,
                    device=device,
                )[:, None]
                masks = (
                    torch.arange(0, max_k, dtype=dynamic_ks.dtype, device=device)[
                        None, :
                    ].expand(num_gt, max_k)
                    < dynamic_ks[:, None]
                )
                _, pos_idxes = torch.topk(cost, k=max_k, dim=1, largest=False)
                pos_idxes.add_(offsets)
                pos_idxes = torch.masked_select(pos_idxes, masks)
                matching_matrix.view(-1).index_fill_(0, pos_idxes, 1)
                del topk_ious, dynamic_ks, pos_idxes, offsets, masks
            else:
                _, pos_idxes = torch.topk(cost, k=max_k, dim=1, largest=False)
                matching_matrix.scatter_(1, pos_idxes, 1)
                del topk_ious, dynamic_ks
        else:
            ks = dynamic_ks.tolist()
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx], k=ks[gt_idx], largest=False)
                matching_matrix[gt_idx][pos_idx] = 1
            del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        anchor_matching_one_more_gt_mask = anchor_matching_gt > 1

        anchor_matching_one_more_gt_inds = torch.nonzero(
            anchor_matching_one_more_gt_mask
        )
        if anchor_matching_one_more_gt_inds.shape[0] > 0:
            anchor_matching_one_more_gt_inds = anchor_matching_one_more_gt_inds[..., 0]
            _, cost_argmin = torch.min(
                cost.index_select(1, anchor_matching_one_more_gt_inds), dim=0
            )
            matching_matrix.index_fill_(1, anchor_matching_one_more_gt_inds, 0)
            matching_matrix[cost_argmin, anchor_matching_one_more_gt_inds] = 1
            fg_mask_inboxes = matching_matrix.any(dim=0)
            fg_mask_inboxes_inds = torch.nonzero(fg_mask_inboxes)[..., 0]
        else:
            fg_mask_inboxes_inds = torch.nonzero(anchor_matching_gt)[..., 0]
        num_fg = fg_mask_inboxes_inds.shape[0]

        matched_gt_inds = matching_matrix.index_select(1, fg_mask_inboxes_inds).argmax(
            0
        )
        fg_mask_inds = fg_mask_inds[fg_mask_inboxes_inds]
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = pair_wise_ious.index_select(
            1, fg_mask_inboxes_inds
        ).gather(
            dim=0, index=matched_gt_inds[None, :]
        )  # [1, matched_gt_inds_count]
        return (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask_inds,
        )

    @staticmethod
    def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, inplace=False):
        if inplace:
            if xyxy:
                tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
                br_hw = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
                br_hw.sub_(tl)
                br_hw.clamp_min_(0)
                del tl
                area_ious = torch.prod(br_hw, 2)
                del br_hw
                area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
                area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
            else:
                tl = torch.max(
                    (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
                )
                br_hw = torch.min(
                    (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
                )
                br_hw.sub_(tl)
                br_hw.clamp_min_(0)
                del tl
                area_ious = torch.prod(br_hw, 2)
                del br_hw
                area_a = torch.prod(bboxes_a[:, 2:], 1)
                area_b = torch.prod(bboxes_b[:, 2:], 1)

            union = area_a[:, None] + area_b - area_ious
            area_ious.div_(union)

            return area_ious
        else:
            if xyxy:
                tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
                br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
                area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
                area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
            else:
                tl = torch.max(
                    (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
                )
                br = torch.min(
                    (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                    (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
                )

                area_a = torch.prod(bboxes_a[:, 2:], 1)
                area_b = torch.prod(bboxes_b[:, 2:], 1)

            hw = (br - tl).clamp(min=0)
            area_i = torch.prod(hw, 2)

            ious = area_i / (area_a[:, None] + area_b - area_i)
            return ious
